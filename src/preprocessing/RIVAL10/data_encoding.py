import json
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.config import PROJECT_ROOT, RIVAL10_CONFIG

def get_filename_mapping(dir_path):
    mapping = {}
    idx = 0

    for filename in os.listdir(dir_path):
        if '.JPEG' in filename and '_mask' not in filename:
            mapping[idx] = filename.replace('.JPEG', '')
            idx += 1

    return mapping

def get_concept_matrix(mapping, dir_path):
    concept_matrix = np.zeros((len(mapping), RIVAL10_CONFIG['N_CONCEPTS']))
    for idx, instance_name in mapping.items():
        concept_path = os.path.join(dir_path, instance_name + '_attr_labels.npy')

        concept_data = np.load(concept_path)
        concept_matrix[idx] = concept_data[0]

    return concept_matrix

def encode_labels_dict(labels_file, mapping):
    with open(labels_file, 'r') as f:
        wnid_data = json.load(f)

    # Extract the first part of the keys in mapping for comparison
    mapping_wnids = {key: value.split('_')[0] for key, value in mapping.items()}
    labels_dict = {}

    for idx, wnid in mapping_wnids.items():
        labels_dict[idx] = wnid_data[wnid]

    return labels_dict

def encode_super_labels_dict(super_labels_file, labels_dict):
    with open(super_labels_file, 'r') as f:
        class_to_super_label = json.load(f)

    super_labels_dict = {}
    for idx, curr_label in labels_dict.items():
        super_labels_dict[idx] = class_to_super_label[curr_label][0]

    return super_labels_dict

def one_hot_encode_labels(labels_file, super_labels_file, mapping):
    labels_dict = encode_labels_dict(labels_file, mapping)
    super_labels_dict = encode_super_labels_dict(super_labels_file, labels_dict)

    labels = list(super_labels_dict.values())

    # Reshape the data to a 2D array as required by OneHotEncoder
    categories_array = np.array(labels).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(categories_array)

    return one_hot_encoded



if __name__ == '__main__':
    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'train', 'ordinary')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'test', 'ordinary')

    # get filename to index mapping
    train_mapping = get_filename_mapping(TRAIN_PATH)
    test_mapping = get_filename_mapping(TEST_PATH)

    # get concept matrices
    train_concept_matrix = get_concept_matrix(train_mapping, TRAIN_PATH)
    test_concept_matrix = get_concept_matrix(test_mapping, TEST_PATH)

    # get image labels
    labels_file = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'meta', 'wnid_to_class.json')
    super_labels_file = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'meta', 'label_mappings.json')

    train_img_labels = one_hot_encode_labels(labels_file, super_labels_file, train_mapping)
    test_img_labels = one_hot_encode_labels(labels_file, super_labels_file, test_mapping)

    print('concepts')
    print(train_concept_matrix.shape)
    print(test_concept_matrix.shape)

    print('labels')
    print(train_img_labels.shape)
    print(test_img_labels.shape)