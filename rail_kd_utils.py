# rail_kd_utils.py

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAILKDLoss(nn.Module):
    def __init__(self, mappings):
        super(RAILKDLoss, self).__init__()
        self.mappings = mappings

    def forward(self, student_outputs, teacher_outputs):
        loss = 0.0
        for s_layer, t_layers in self.mappings.items():
            for s_block, t_block in t_layers:
                student_output = student_outputs[s_layer][s_block]
                teacher_output = teacher_outputs[s_layer][t_block]
                loss += F.mse_loss(student_output, teacher_output)
        return loss


def generate_sorted_random_mappings(student_blocks, teacher_blocks):
    mappings = {}
    for s_layer in student_blocks.keys():
        s_blocks = student_blocks[s_layer]
        t_blocks = teacher_blocks[s_layer]
        if len(s_blocks) > len(t_blocks):
            raise ValueError("Teacher blocks should be greater than or equal to student blocks in each layer.")
        t_indices = sorted(random.sample(range(len(t_blocks)), len(s_blocks)))
        mappings[s_layer] = [(i, t_indices[i]) for i in range(len(s_blocks))]
    return mappings


def set_forward_hook(layer, outputs_dict, layer_name):
    def hook_fn(module, input, output):
        outputs_dict[layer_name].append(output)
    layer.register_forward_hook(hook_fn)

# Define the block mappings
student_blocks = {
    'layer1': [0, 1],
    'layer2': [0, 1],
    'layer3': [0, 1],
    'layer4': [0, 1]
}

teacher_blocks = {
    'layer1': list(range(3)),
    'layer2': list(range(13)),
    'layer3': list(range(30)),
    'layer4': list(range(3))
}

random_sorted_mappings = generate_sorted_random_mappings(student_blocks, teacher_blocks)
