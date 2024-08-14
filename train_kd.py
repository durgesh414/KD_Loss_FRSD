import os
import yaml
import time
import argparse
import torch

from datetime import datetime
from runner_stat_kd import IterRunner
from utils import fill_config
from builder import build_dataloader, build_model


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--student_config', help='student config file path')
    parser.add_argument('--teacher_config', help='teacher config file path')
    parser.add_argument('--start_time', help='time to start training')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.student_config, 'r') as f:
        student_config = yaml.load(f, yaml.SafeLoader)
    
    with open(args.teacher_config, 'r') as f:
        teacher_config = yaml.load(f, yaml.SafeLoader)

    student_config['data'] = fill_config(student_config['data'])
    student_config['model'] = fill_config(student_config['model'])

    teacher_config['data'] = fill_config(teacher_config['data'])
    teacher_config['model'] = fill_config(teacher_config['model'])

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified. CPU mode is not supported yet')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader = build_dataloader(student_config['data']['train'])
    val_loaders = build_dataloader(student_config['data']['val'])

    torch.cuda.set_device(0)
    num_class = len(train_loader.dataset.classes)
    
    # init student model
    student_feat_dim = student_config['model']['backbone']['net']['out_channel']
    student_config['model']['head']['net']['feat_dim'] = student_feat_dim
    student_config['model']['head']['net']['num_class'] = num_class
    student_model = build_model(student_config['model'])
    
    # init teacher model
    teacher_feat_dim = teacher_config['model']['backbone']['net']['out_channel']
    teacher_config['model']['head']['net']['feat_dim'] = teacher_feat_dim
    teacher_config['model']['head']['net']['num_class'] = num_class
    teacher_model = build_model(teacher_config['model'])

    runner = IterRunner(student_config, teacher_config, train_loader, val_loaders, student_model, teacher_model)
    runner.run()

if __name__ == '__main__':
    main()

