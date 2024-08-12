import argparse
import numpy as np
import json
from utils.drdo_evaluations_1 import evaluate
from utils.drdo_utils import calculate_sum, cos_distance


classs = "neg"

def load_image_pairs(pairs_file):
    pairs = np.load(pairs_file, allow_pickle=True)
    print("pairs shape", pairs.shape)
    return pairs

def load_embeddings(embeddings_file):
    with open(embeddings_file, 'r') as f:
        embedding_cache = json.load(f)
    return embedding_cache

def inference(embedding_cache, pairs):
    print("[*] Perform Evaluation...")

    batch_size = 50000
    result_list = []
    epoch = "Null"

    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        embeddings = []
        issame_list = []

        for pair in batch_pairs:
            img_path1, img_path2 = pair
            img_path1 = img_path1.replace("/", "#")
            img_path2 = img_path2.replace("/", "#")

            embedding1 = embedding_cache.get(img_path1)
            embedding2 = embedding_cache.get(img_path2)

            if embedding1 is None or embedding2 is None:
                print(f"Warning: Could not find embeddings for {img_path1} or {img_path2}")
                continue

            embeddings.append(embedding1)
            embeddings.append(embedding2)

            if classs == "pos":
                issame_list.append(1)  
            else:
                issame_list.append(0) 

        embeddings = np.array(embeddings)
        issame = np.array(issame_list)
        print("emb shape & issame shape ", embeddings.shape, issame.shape)
   
        results = evaluate(embeddings, issame)
        print(results)
        result_list.append(results)

    print("Len", len(result_list))
    store_dict, sum_dict = calculate_sum(result_list)
    print("Results, Store_dict: ", store_dict)
    print("Sum_dict: " , sum_dict)

    # Convert numpy.int64 to int for JSON serialization
    sum_dict_serializable = {key: {k: int(v) if isinstance(v, np.int64) else v for k, v in value.items()} for key, value in sum_dict.items()}


    with open(f'saved_results/triplet_arcface_18_results_L2_distance/cosine/5m_{classs}.json', 'w') as json_file: #{epoch}/
        json.dump(sum_dict_serializable, json_file)
    print(f"Doneeee!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Pairs Inference')
    parser.add_argument('--embeddings_file', type=str, default=None, help='Path to the JSON file containing precomputed embeddings')
    args = parser.parse_args()

    # Load embeddings and image pairs
    embedding_cache = load_embeddings(args.embeddings_file)
    pairs = load_image_pairs(f"/home/kaushik/Durgesh/drdo-faces/final_pairs/og_all_5m/{classs}all.npy")
    inference(embedding_cache, pairs)
