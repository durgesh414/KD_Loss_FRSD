import os
import yaml
import time
import argparse
import torch

from datetime import datetime
from runner import IterRunner
from utils import fill_config
from builder import build_dataloader, build_model


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', help='train config file path')
    #parser.add_argument('--proj_dir', help='the dir to save logs and models')
    parser.add_argument('--start_time', help='time to start training')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    config['data'] = fill_config(config['data'])
    config['model'] = fill_config(config['model'])

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                'CPU mode is not supported yet')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(config['data']['train'])
    val_loaders = build_dataloader(config['data']['val'])

    # init model
    torch.cuda.set_device(0)
    feat_dim = config['model']['backbone']['net']['out_channel']
    config['model']['head']['net']['feat_dim'] = feat_dim
    num_class = len(train_loader.dataset.classes)
    config['model']['head']['net']['num_class'] = num_class

    model = build_model(config['model'])

    runner = IterRunner(config, train_loader, val_loaders, model)
    runner.run()

if __name__ == '__main__':
    main()

