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
        self.cos_sim_loss_weight = self.student_config['model']["head"]['cos_sim_loss_weight'] 
        self.attention_maps_weight = self.student_config['model']["head"]['attention_maps_weight'] 
        self.sim_matrix_weight = self.student_config['model']["head"]['sim_matrix_weight'] 
        print("cosine_weight & sim matrix weight", self.cos_sim_loss_weight, self.sim_matrix_weight)

        if self.rank != 0:
            return

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Loading state dicts for student and teacher models
        # student_state_dict_path = "project/og/r18_og_vggface2_20240617_174438/models/backbone_140000.pth"
        teacher_state_dict_path = "project/og/r100_og_vggface2_20240617_174425/models/backbone_140000.pth"

        # self.student_model['backbone']['net'].load_state_dict(torch.load(student_state_dict_path))
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

        # Initialize memory banks for storing features
        self.memory_bank_student = None
        self.memory_bank_teacher = None
        self.memory_bank_size = 3 * train_loader.batch_size


    def update_memory_bank(self, new_student_feats, new_teacher_feats):
        # Detach the features before adding them to the memory bank
        new_student_feats = new_student_feats.detach()
        new_teacher_feats = new_teacher_feats.detach()

        # If memory bank is not initialized, do it now
        if self.memory_bank_student is None:
            self.memory_bank_student = new_student_feats
            self.memory_bank_teacher = new_teacher_feats
        else:
            # Concatenate new features to the existing memory bank
            self.memory_bank_student = torch.cat((self.memory_bank_student, new_student_feats), dim=0)
            self.memory_bank_teacher = torch.cat((self.memory_bank_teacher, new_teacher_feats), dim=0)

            # If the memory bank exceeds the allowed size, dequeue the oldest features
            if self.memory_bank_student.size(0) > self.memory_bank_size:
                self.memory_bank_student = self.memory_bank_student[-self.memory_bank_size:]
                self.memory_bank_teacher = self.memory_bank_teacher[-self.memory_bank_size:]


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
        data, labels = next(self.train_loader)
        data, labels = data.to(self.rank), labels.to(self.rank)

        # Forward pass
        self.set_model(test_mode=False)
        student_feats = self.student_model['backbone']['net'](data)
        student_loss = self.student_model['head']['net'](student_feats, labels)
        teacher_feats = self.teacher_model['backbone']['net'](data)

        # Update the memory bank with current features
        self.update_memory_bank(student_feats, teacher_feats)

        # Only calculate similarity loss if memory bank is sufficiently populated
        if self.memory_bank_student.size(0) >= self.memory_bank_size:
            # Use features from the memory bank for similarity computation
            student_feats_bank = self.memory_bank_student
            teacher_feats_bank = self.memory_bank_teacher

            # Compute cosine similarity matrices for both teacher and student features
            teacher_similarity_matrix = F.cosine_similarity(teacher_feats_bank.unsqueeze(1), teacher_feats_bank.unsqueeze(0), dim=-1)
            student_similarity_matrix = F.cosine_similarity(student_feats_bank.unsqueeze(1), student_feats_bank.unsqueeze(0), dim=-1)

            similarity_diff = abs(teacher_similarity_matrix - student_similarity_matrix)

            # Create a mask for the upper triangular part, including the diagonal
            mask = torch.triu(torch.ones_like(similarity_diff), diagonal=0)
            similarity_diff_upper = similarity_diff * mask

            # Parameters for the transformation
            r = 20. 
            m = 0.2
            # Implementing the logarithmic transformation
            transformed_similarity_diff_upper = (1 / r) * torch.log(1 + torch.exp(r * (similarity_diff_upper - m)))
    
            # shift = 0.1
            # transformed_similarity_diff_upper = similarity_diff_upper / (similarity_diff_upper + shift)

            # Compute the similarity loss by taking the sum
            similarity_loss_sum = torch.sum(transformed_similarity_diff_upper)

            # Assuming similarity_diff_upper is an upper triangular matrix
            n = similarity_diff_upper.shape[0]
            num_upper_tri_elements = ((n * (n - 1)) / 2 ) - n

            # Normalizing the loss by the number of elements in the upper triangular part
            similarity_loss = similarity_loss_sum / num_upper_tri_elements


            # # Calculate the Frobenius norm of the upper triangular matrix
            # frobenius_norm = torch.norm(transformed_similarity_diff_upper, p='fro')
            # print("***", frobenius_norm)
           
            # print("mean:", torch.mean(transformed_similarity_diff_upper), "sum:", similarity_loss_sum, "Loss:", similarity_loss, "const_scale: ", similarity_loss_sum/num_upper_tri_elements)

            # # Compute the similarity loss
            # similarity_loss = torch.mean(transformed_similarity_diff_upper)
            kd_loss = self.sim_matrix_weight * similarity_loss
        else:
            kd_loss = torch.tensor(0.0).to(self.rank)

        total_student_loss = student_loss + kd_loss
        total_loss = total_student_loss
        total_loss.backward()

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

        self.update_model()

        if self.rank == 0:
            magnitude = torch.norm(student_feats, 2, 1)
            msg = {
                'Iter': self._iter,
                'Student_Loss': student_loss.item(),
                'KD_Loss': kd_loss.item(),
                'Total_Student_Loss': total_student_loss.item(),
                'Mag_mean': magnitude.mean().item(),
                'Mag_std': magnitude.std().item(),
                'bkb_grad': b_grad,
                'head_grad': h_grad,
                'sym_matrix': similarity_loss_sum.item() if self.memory_bank_student.size(0) >= self.memory_bank_size else 0.0
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
        # self.freeze_layers(self.student_model['backbone']['net'])

        while self._iter <= self._max_iters:
            if self._iter % self.val_intvl == 0 and self._iter > 0:
                self.val()

            if self._iter in self.save_iters and self.rank == 0:
                self.save_model()

            self.train()
            self._iter += 1
