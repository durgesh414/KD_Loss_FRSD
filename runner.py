import os
import os.path as osp
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import IterLoader, LoggerBuffer
from torch.nn.utils import clip_grad_norm_

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings("ignore", message="Leaking Caffe2 thread-pool after fork.")


class IterRunner():

    def __init__(self, config, train_loader, val_loaders, model):
        
        self.config = config
        self.train_loader = IterLoader(train_loader)
        self.val_loaders = val_loaders
        self.model = model
        self.rank = 0
        self.world_size = 1

        # meta variables of a new runner
        proj_cfg = config['project']
        self._iter = 0
        self._max_iters = [max(cfg['scheduler']['milestones']) for cfg in config['model'].values()]
        self._max_iters = max(self._max_iters)      # Max number of iters 
        self.val_intvl = proj_cfg['val_intvl']      # Saving val info in log file
        self.save_iters = proj_cfg['save_iters']    # Saving training info in log file
       
        if self.rank != 0:
            return

        # Loading state dicts for student and teacher models
        model_state_dict_path = "project/r100_og_vggface2_20240617_174425/models/backbone_140000.pth"
        # "project/sf2_resnet100_og_vggface2_20240617_174425/models/backbone_140000.pth"
        # "project/sf2_resnet18_og_vggface2_20240617_174438/models/backbone_140000.pth"
        self.model['backbone']['net'].load_state_dict(torch.load(model_state_dict_path))

        # Freeze the teacher model parameters
        for param in self.model['backbone']['net'].parameters():
            param.requires_grad = False
    
        # project directory
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        proj_dir = proj_cfg['proj_dir']
        proj_dir = osp.join(proj_dir, timestamp)
        if not osp.exists(proj_dir):
            os.makedirs(proj_dir)
        proj_cfg['proj_dir'] = proj_dir
        print('')
        print('The training log and models are saved to the og ' + proj_dir)
        print('')

        # model directory
        self.model_dir = osp.join(proj_dir, proj_cfg['model_dir'])
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)
        proj_cfg['model_dir'] = self.model_dir

        # logger
        train_log_cfg = proj_cfg['train_log']
        train_log_cfg['path'] = osp.join(proj_dir, train_log_cfg['path'])
        self.train_buffer = LoggerBuffer(name='train', **train_log_cfg)

        val_log_cfg = proj_cfg['val_log']
        val_log_cfg['path'] = osp.join(proj_dir, val_log_cfg['path'])
        self.val_buffer = LoggerBuffer(name='val', **val_log_cfg)

        # save config to proj_dir
        config_path = osp.join(proj_dir, proj_cfg['cfg_fname'])
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)


    def freeze_layers(self, model):
        """
        Freeze all layers up to and including the second block of layer4.
        Unfreeze all layers after that.
        """
        freeze = True
        for name, module in model.named_modules():
            if  "layer4" in name:  # "layer3" in name or
                freeze = False  # Start unfreezing after layer4's second block
            for param in module.parameters():
                param.requires_grad = not freeze
            if freeze:
                print(f"Froze {name}")
            else:
                print(f"Unfroze {name}")

        # Additionally unfreeze layers after layer4
        for name, module in model.named_children():
            if name in ['bn2', 'dropout', 'fc', 'features']:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze {name}")
        

    def set_model(self, test_mode):
        for module in self.model:
            if test_mode:
                self.model[module]['net'].eval()
            else:
                self.model[module]['net'].train()
                self.model[module]['optimizer'].zero_grad()


    def update_model(self):
        lrs = []
        for module in self.model:
            self.model[module]['optimizer'].step()
            self.model[module]['scheduler'].step()
            lrs.extend(self.model[module]['scheduler'].get_last_lr())

        if getattr(self, 'current_lrs', None) != lrs and self.rank == 0:
            self.current_lrs = lrs
            lr_msg = ', '.join(
                    ['{:3.5f}'.format(lr) for lr in self.current_lrs])
            self.train_buffer.logger.info(
                    'Lrs are changed to {}'.format(lr_msg))


    def save_model(self):
        for module in self.model:
            model_name = '{}_{}.pth'.format(str(module), str(self._iter))
            model_path = osp.join(self.model_dir, model_name)
            torch.save(self.model[module]['net'].state_dict(), model_path)


    def train(self):
        data, labels = next(self.train_loader)
        data, labels = data.to(self.rank), labels.to(self.rank)

        # forward
        self.set_model(test_mode=False)
        feats = self.model['backbone']['net'](data)       #batch_size x 512

        loss = self.model['head']['net'](feats, labels)   #SphereFace2()
        
        # backward
        loss.backward()
        b_norm = self.model['backbone']['clip_grad_norm']
        h_norm = self.model['head']['clip_grad_norm']
        if b_norm < 0. or h_norm < 0.:
            raise ValueError(
                    'the clip_grad_norm should be positive. ({:3.4f}, {:3.4f})'.format(b_norm, h_norm))
         
        b_grad = clip_grad_norm_(
                self.model['backbone']['net'].parameters(),
                max_norm=b_norm, norm_type=2)
        h_grad = clip_grad_norm_(
                self.model['head']['net'].parameters(),
                max_norm=h_norm, norm_type=2)

        # update model
        self.update_model()

        if self.rank == 0:
            # logging and update meters
            magnitude = torch.norm(feats, 2, 1) 
            msg = {
                'Iter': self._iter,
                'Loss': loss.item(),
                #'MHE_Loss': mhe_loss.item(),  # Log the MHE loss
                'Mag_mean': magnitude.mean().item(),
                'Mag_std': magnitude.std().item(),
                'bkb_grad': b_grad,
                'head_grad': h_grad,
            }
            self.train_buffer.update(msg)

    
    @torch.no_grad()
    def val(self):
        # switch to test mode
        self.set_model(test_mode=True)
        msg = {'Iter': self._iter}

        for val_loader in self.val_loaders:
            # meta info
            dataset = val_loader.dataset
            # create a placeholder `feats`,
            # compute _feats in different GPUs and collect 
            dim = self.config['model']['backbone']['net']['out_channel']
            feats = torch.zeros(
                    [len(dataset), dim], dtype=torch.float32).to(self.rank)
            for data, indices in val_loader:
                data = data.to(self.rank)
                _feats = self.model['backbone']['net'](data)
                data = torch.flip(data, [3])
                _feats += self.model['backbone']['net'](data)
                feats[indices, :] = _feats

            # dist.all_reduce(feats, op=dist.ReduceOp.SUM)
            results = dataset.evaluate(feats.cpu())
            results = dict(results)
            metric = val_loader.dataset.metrics[0]
            msg[dataset.name] = results[metric]

        if self.rank == 0:
            self.val_buffer.update(msg)


    def run(self):
        self.freeze_layers(self.model['backbone']['net'])

        while self._iter <= self._max_iters:

            # train step
            if self._iter % self.val_intvl == 0 and self._iter > 0:
                self.val()

            if self._iter in self.save_iters and self.rank == 0:
                self.save_model()

            self.train()
            self._iter += 1

