import os
import os.path as osp
import yaml
import argparse
import numpy as np
import cv2
import json
import torch
import torch.nn as nn
import logging

from utils import fill_config
from builder import build_dataloader, build_from_cfg

def load_image_pairs(pairs_file):
    return np.load(pairs_file, allow_pickle=True)

def parse_args():
    parser = argparse.ArgumentParser(description='A PyTorch project for face recognition with embeddings caching.')
    parser.add_argument('--model_name', type=str, default=None, help='Model Name')
    parser.add_argument('--proj_dirs', nargs='+', help='The project directories to be tested', required=True)
    return parser.parse_args()

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Could not read image at {img_path}")
        return None
        

    img = img / 255.0  # Normalize
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add batch dimension
    return img_tensor

@torch.no_grad()
def extract_embeddings(model, pairs, image_folder):
    embedding_cache = {}

    for pair in pairs:
        for img_name in pair:
            img_name = img_name.replace("/", "#")
            if img_name in embedding_cache:
                continue

            img_full_path = osp.join(image_folder, img_name)
            img_tensor = preprocess_image(img_full_path)
            if img_tensor is None:
                continue

            img_tensor = img_tensor.cuda()
            embedding = model(img_tensor).squeeze().cpu().numpy().tolist()
            embedding_cache[img_name] = embedding
    return embedding_cache

def main_worker(model_name, class_name, proj_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folder = "/home/kaushik/Durgesh/drdo-faces/og_all_5m" 

    logging.info(f"Evaluating project: {proj_dir}")

    config_path = osp.join(proj_dir, "config.yml")  # YAML file of student 

    with open(config_path, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.SafeLoader)

    bkb_net = build_from_cfg(test_config['model']['backbone']['net'], 'model.backbone').to(device)
    bkb_net.eval()
    
    save_iters = [50000, 100000] #[120000, 140000]

    for save_iter in save_iters:
        weight_path = osp.join(proj_dir, "models", f'backbone_{save_iter}.pth')
        bkb_net.load_state_dict(torch.load(weight_path, map_location=device))

        pairs = load_image_pairs(f"/home/kaushik/Durgesh/drdo-faces/final_pairs/og_all_5m/{class_name}all.npy")
        embedding_cache = extract_embeddings(bkb_net, pairs, image_folder)

        # Save embedding cache
        cache_dir = f"drdo/cache/og/{model_name}_{save_iter}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"embedding_cache_{model_name}.json")
            
        with open(cache_path, 'w') as f:
            json.dump(embedding_cache, f, indent=4)

        logging.info(f"Embedding cache saved to {cache_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    model_name = args.model_name
    class_name = "pos"
    proj_dirs = args.proj_dirs

    for proj_dir in proj_dirs:
        main_worker(model_name, class_name, proj_dir)



# import os
# import os.path as osp
# import yaml
# import argparse
# import numpy as np
# import cv2
# import json
# import torch
# import torch.nn as nn

# from utils import fill_config
# from builder import build_dataloader, build_from_cfg

# def load_image_pairs(pairs_file):
#     return np.load(pairs_file, allow_pickle=True)

# def parse_args():
#     parser = argparse.ArgumentParser(description='A PyTorch project for face recognition with embeddings caching.')
#     parser.add_argument('--model_name', type=str, default=None, help='Model Nameee')
#     parser.add_argument('--proj_dirs', '--list', nargs='+', help='the project directories to be tested', required=True)
#     return parser.parse_args()


# def preprocess_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Warning: Could not read image at {img_path}")
#         return None

#     #print(f"Yeeee Could read image at {img_path}")
#     img = img / 255.0  # Normalize
#     img = img[:, :, ::-1]  # BGR to RGB
#     img = np.transpose(img, (2, 0, 1)).copy()
#     img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add batch dimension
#     return img_tensor


# @torch.no_grad()
# def extract_embeddings(model, pairs, image_folder):
#     embedding_cache = {}

#     for pair in pairs:
#         for img_name in pair:
#             img_name = img_name.replace("/", "#")
#             if img_name in embedding_cache:
#                 continue

#             img_full_path = osp.join(image_folder, img_name)
#             img_tensor = preprocess_image(img_full_path)
#             if img_tensor is None:
#                 continue

#             img_tensor = img_tensor.cuda()
#             embedding = model(img_tensor).squeeze().cpu().numpy().tolist()
#             embedding_cache[img_name] = embedding
#     return embedding_cache



# def main_worker(model_namee, classs, proj_dir):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image_folder = "/home/kaushik/Durgesh/drdo-faces/og_all_5m" 

#     # print(f"Evaluating project: {proj_dir}")

#     config_path = osp.join(proj_dir, "config.yml")  # yaml file of student 

#     with open(config_path, 'r') as f:
#         test_config = yaml.load(f, Loader=yaml.SafeLoader)

#     #print("************", test_config['model']['backbone']['net'])
#     bkb_net = build_from_cfg(test_config['model']['backbone']['net'], 'model.backbone').to(device)
#     bkb_net.eval()
    
#     save_iters = [120000, 140000]

#     for save_iter in save_iters:
#         weight_path = osp.join(proj_dir, "models", f'backbone_{save_iter}.pth')
#         bkb_net.load_state_dict(torch.load(weight_path, map_location=device))

#         pairs = load_image_pairs(f"/home/kaushik/Durgesh/drdo-faces/final_pairs/og_all_5m/{classs}all.npy")
#         embedding_cache = extract_embeddings(bkb_net, pairs, image_folder)

#         # Save embedding cache
#         cache_dir = f"drdo/cache/kd/{model_namee}_{save_iter}"
#         os.makedirs(cache_dir, exist_ok=True)
#         cache_path = osp.join(cache_dir, f"embedding_cache_{model_namee}.json")
            
#         with open(cache_path, 'w') as f:
#             json.dump(embedding_cache, f, indent=4)

#         print(f"Embedding cache saved to {cache_path}")


# if __name__ == '__main__':
#     args = parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.load(f, Loader=yaml.SafeLoader)

#     model_namee = args.model_name
#     classs = "neg"
#     proj_dir = args.proj_dirs

#     main_worker(model_namee, classs, proj_dir)
