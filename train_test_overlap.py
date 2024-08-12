import os
import os.path as osp
import numpy as np
import re

def load_image_pairs(pairs_file):
    return np.load(pairs_file, allow_pickle=True)

def extract_class_names(pairs):
    class_names = []
    for row in pairs:
        classs = row[0][53:-4]
        class_names.append(classs)
    return list(set(class_names))

def extract_class_names_from_txt(file_path):
    class_names = set()
    with open(file_path, 'r') as file:
        for line in file:
            class_names.add(line[13:18])
    return list(class_names)

def get_test_class_names(class_name):
    pairs = load_image_pairs(f"/home/kaushik/Durgesh/drdo-faces/final_pairs/og_all_5m/{class_name}all.npy")
    return extract_class_names(pairs)



def write_train_classes_to_file(image_folder, test_class_names, output_file):
    with open(output_file, 'w') as file:
        for root, _, files in os.walk(image_folder):
            for name in files:
                # print(name)
                pass
                # if name.endswith('.jpg'):
                #     match = re.search(r'DLORD_(\d{5})', name)
                #     print(match)
                #     if match:
                #         class_name = match.group(1)
                #         if class_name not in test_class_names:
                #             file.write(f"{os.path.join(root, name)} DLORD_{class_name}\n")




def main_worker(class_name):
    image_folder = "/home/kaushik/Durgesh/drdo-faces/og_all_5m"
    test_class_names = get_test_class_names(class_name)
    print("Test class names:", test_class_names)
    print("Number of test class names:", len(test_class_names))

    file_path = '/home/kaushik/Durgesh/data/dlord_train_112x112.txt'
    train_class_names = extract_class_names_from_txt(file_path)
    # print("Train class names:", train_class_names)
    print("Number of train class names:", len(train_class_names))

    # Find the overlap between train and test class names
    overlap = set(train_class_names).intersection(set(test_class_names))
    # print("Overlap between train and test class names:", overlap)
    print("Number of overlapping class names:", len(overlap))

    # Write the disjoint training set to a file
    output_file = '/home/kaushik/Durgesh/data/dlord_train_disjoint.txt'
    write_train_classes_to_file(image_folder, test_class_names, output_file)
    print(f"Training data written to {output_file}")

    test_class_names_array = np.array(test_class_names)
    np.save('/home/kaushik/Durgesh/data/test_class_names_array.npy', test_class_names_array)


if __name__ == '__main__':
    class_name = "pos"
    main_worker(class_name)
















# import os
# import os.path as osp
# import yaml
# import numpy as np
# import cv2
# import json
# import re

# def load_image_pairs(pairs_file):
#     return np.load(pairs_file, allow_pickle=True)

# def extract_class_names(pairs):
#     class_names = []
#     for row in pairs:
#         classs = row[0][53:-4]
#         class_names.append(classs)
#     return list(set(class_names))


# def extract_class_names_from_txt(file_path):
#     class_names = set()
#     with open(file_path, 'r') as file:
#         for line in file:
#             # print(line[13:18])   
#             class_names.add(line[13:18])
#     return list(class_names)




# def main_worker(class_name):
#     image_folder = "/home/kaushik/Durgesh/drdo-faces/og_all_5m" 
#     pairs = load_image_pairs(f"/home/kaushik/Durgesh/drdo-faces/final_pairs/og_all_5m/{class_name}all.npy")
    
#     test_class_names = extract_class_names(pairs)
#     print(test_class_names)
#     print(len(test_class_names))

#     file_path = '/home/kaushik/Durgesh/data/dlord_train_112x112.txt'
#     train_class_names = extract_class_names_from_txt(file_path)
#     print(train_class_names)
#     print(len(train_class_names))

#     # Find the overlap between train and test class names
#     overlap = set(train_class_names).intersection(set(test_class_names))
#     print("Overlap between train and test class names:", overlap)
#     print("Number of overlapping class names:", len(overlap))

# if __name__ == '__main__':
#     class_name = "pos"
#     main_worker(class_name)