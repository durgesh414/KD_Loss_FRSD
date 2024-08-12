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

def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output
        #print(f"{name} activation shape: {output.shape}")
    return hook


def compute_attention_map(activation):
    # Sum of absolute values of the activations
    # print(activation.shape)
    attention_map = torch.sum(torch.abs(activation), dim=1, keepdim=True)
    
    # Normalize the attention map
    attention_map = attention_map / attention_map.max()
    
    return attention_map


class IterRunner():

    def __init__(self, student_config, teacher_config, train_loader, val_loaders, student_model, teacher_model):
        self.student_config = student_config
        self.teacher_config = teacher_config
        self.train_loader = IterLoader(train_loader)
        self.val_loaders = val_loaders
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.rank = 0
        self.world_size = 1

        student_proj_cfg = student_config['project']
        teacher_proj_cfg = teacher_config['project']
        self._iter = 0
        self._max_iters = max([max(cfg['scheduler']['milestones']) for cfg in student_config['model'].values()])

        self.val_intvl = student_proj_cfg['val_intvl']
        self.save_iters = student_proj_cfg['save_iters']

        # Combine the original loss and cosine similarity loss
        self.kl_div_loss_weight = 1. #self.student_config['model']["head"]['cos_sim_loss_weight'] 

        print("KL Div", self.kl_div_loss_weight)

        if self.rank != 0:
            return

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Loading state dicts for student and teacher models
        student_state_dict_path = "project/r18_og_vggface2_20240617_174438/models/backbone_140000.pth"
        teacher_state_dict_path = "project/r100_og_vggface2_20240617_174425/models/backbone_140000.pth"

        self.student_model['backbone']['net'].load_state_dict(torch.load(student_state_dict_path))
        self.teacher_model['backbone']['net'].load_state_dict(torch.load(teacher_state_dict_path))

        # Freeze the teacher model parameters
        for param in self.teacher_model['backbone']['net'].parameters():
            param.requires_grad = False

        student_proj_dir = osp.join(student_proj_cfg['proj_dir'], timestamp)
        if not osp.exists(student_proj_dir):
            os.makedirs(student_proj_dir)
        student_proj_cfg['proj_dir'] = student_proj_dir
        print(f'\nThe training log and models are saved to {student_proj_dir}\n')

        self.student_model_dir = osp.join(student_proj_dir, student_proj_cfg['model_dir'])
        if not osp.exists(self.student_model_dir):
            os.makedirs(self.student_model_dir)
        student_proj_cfg['model_dir'] = self.student_model_dir
        self.teacher_model_dir = osp.join(student_proj_dir, teacher_proj_cfg['model_dir'])
        if not osp.exists(self.teacher_model_dir):
            os.makedirs(self.teacher_model_dir)
        teacher_proj_cfg['model_dir'] = self.teacher_model_dir

        #train log
        train_log_cfg = student_proj_cfg['train_log']
        train_log_cfg['path'] = osp.join(student_proj_dir, train_log_cfg['path'])
        self.train_buffer = LoggerBuffer(name='train', **train_log_cfg)

        #val log
        student_val_log_cfg = student_proj_cfg['val_log']
        student_val_log_cfg['path'] = osp.join(student_proj_dir, 'student_' + student_val_log_cfg['path'])
        self.student_val_buffer = LoggerBuffer(name='student_val', **student_val_log_cfg)

        teacher_val_log_cfg = teacher_proj_cfg['val_log']
        teacher_val_log_cfg['path'] = osp.join(student_proj_dir, 'teacher_' + teacher_val_log_cfg['path'])
        self.teacher_val_buffer = LoggerBuffer(name='teacher_val', **teacher_val_log_cfg)

        #config file
        student_config_path = osp.join(student_proj_dir, student_proj_cfg['cfg_fname'])
        with open(student_config_path, 'w') as f:
            yaml.dump(student_config, f, sort_keys=False, default_flow_style=None)

        teacher_config_path = osp.join(student_proj_dir, teacher_proj_cfg['cfg_fname'])
        with open(teacher_config_path, 'w') as f:
            yaml.dump(teacher_config, f, sort_keys=False, default_flow_style=None)


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
        for module in self.student_model:
            if test_mode:
                self.student_model[module]['net'].eval()
            else:
                self.student_model[module]['net'].train()
                self.student_model[module]['optimizer'].zero_grad()

        # Always keeping the teacher model in evaluation mode
        for module in self.teacher_model:
            self.teacher_model[module]['net'].eval()
        

    def update_model(self):
        lrs = []
        for module in self.student_model:
            self.student_model[module]['optimizer'].step()
            self.student_model[module]['scheduler'].step()
            lrs.extend(self.student_model[module]['scheduler'].get_last_lr())

        if getattr(self, 'current_lrs', None) != lrs and self.rank == 0:
            self.current_lrs = lrs
            lr_msg = ', '.join(
                ['{:3.5f}'.format(lr) for lr in self.current_lrs])
            self.train_buffer.logger.info(
                'Lrs are changed to {}'.format(lr_msg))


    def save_model(self):
        for module in self.student_model:
            model_name = 'student_{}_{}.pth'.format(str(module), str(self._iter))
            model_path = osp.join(self.student_model_dir, model_name)
            torch.save(self.student_model[module]['net'].state_dict(), model_path)


    def train(self):
        # Get a batch of data and labels
        data, labels = next(self.train_loader)
        data, labels = data.to(self.rank), labels.to(self.rank)

        # Forward pass for the student model
        self.set_model(test_mode=False)
        student_logits = self.student_model['backbone']['net'](data)
        student_loss = self.student_model['head']['net'](student_logits, labels)

        # Forward pass for the teacher model (teacher model is in eval mode)
        with torch.no_grad():
            teacher_logits = self.teacher_model['backbone']['net'](data)


        # Apply temperature scaling to the logits
        temperature = 3.0 #self.student_config['model']["head"].get('temperature', 3.0)
        student_soft_logits = student_logits / temperature
        teacher_soft_logits = teacher_logits / temperature


        # Compute softmax probabilities
        student_probs = F.log_softmax(student_soft_logits, dim=1)
        teacher_probs = F.softmax(teacher_soft_logits, dim=1)


        # Compute KL divergence loss
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        weighted_kl_loss = self.kl_div_loss_weight * kl_loss
        total_loss =  student_loss + weighted_kl_loss


        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        b_norm = self.student_model['backbone']['clip_grad_norm']
        h_norm = self.student_model['head']['clip_grad_norm']
        if b_norm < 0. or h_norm < 0.:
            raise ValueError(
                'the clip_grad_norm should be positive. ({:3.4f}, {:3.4f})'.format(b_norm, h_norm))

        b_grad = clip_grad_norm_(
            self.student_model['backbone']['net'].parameters(),
            max_norm=b_norm, norm_type=2)
        h_grad = clip_grad_norm_(
            self.student_model['head']['net'].parameters(),
            max_norm=h_norm, norm_type=2)

        # Update the model parameters
        self.update_model()

        # Logging the loss and other statistics
        if self.rank == 0:
            magnitude = torch.norm(student_logits, 2, 1)
            msg = {
                'Iter': self._iter,
                'Student_Loss': student_loss.item(),
                'KD_Loss': weighted_kl_loss.item(),
                'Total_Student_Loss': total_loss.item(),
                'Mag_mean': magnitude.mean().item(),
                'Mag_std': magnitude.std().item(),
                'bkb_grad': b_grad,
                'head_grad': h_grad,
            }
            self.train_buffer.update(msg)


    @torch.no_grad()
    def val(self):
        self.set_model(test_mode=True)
        student_msg = {'Iter': self._iter}
        teacher_msg = {'Iter': self._iter}

        for val_loader in self.val_loaders:
            dataset = val_loader.dataset
            dim = self.student_config['model']['backbone']['net']['out_channel']

            # Placeholders for student and teacher features
            student_feats = torch.zeros([len(dataset), dim], dtype=torch.float32).to(self.rank)
            teacher_feats = torch.zeros([len(dataset), dim], dtype=torch.float32).to(self.rank)

            # Iterate over the validation loader
            for data, indices in val_loader:
                data = data.to(self.rank)

                _student_feats = self.student_model['backbone']['net'](data)
                _teacher_feats = self.teacher_model['backbone']['net'](data)

                data = torch.flip(data, [3])

                _student_feats += self.student_model['backbone']['net'](data)
                student_feats[indices, :] = _student_feats

                _teacher_feats += self.teacher_model['backbone']['net'](data)
                teacher_feats[indices, :] = _teacher_feats

            # Evaluate the features
            student_results = dataset.evaluate(student_feats.cpu())
            student_results = dict(student_results)

            teacher_results = dataset.evaluate(teacher_feats.cpu())
            teacher_results = dict(teacher_results)

            metric = val_loader.dataset.metrics[0]
            student_msg[dataset.name] = student_results[metric]
            teacher_msg[dataset.name] = teacher_results[metric]

        if self.rank == 0:
            self.student_val_buffer.update(student_msg)
            self.teacher_val_buffer.update(teacher_msg)

    def run(self):
        self.freeze_layers(self.student_model['backbone']['net'])

        while self._iter <= self._max_iters:
            if self._iter % self.val_intvl == 0 and self._iter > 0:
                self.val()

            if self._iter in self.save_iters and self.rank == 0:
                self.save_model()

            self.train()
            self._iter += 1

