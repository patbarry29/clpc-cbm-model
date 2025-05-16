import os
import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT
from derm7pt.dataset import Derm7PtDataset


def vprint(message, is_verbose):
    if is_verbose:
        print(message)

def get_filename_to_id_mapping(filepath, reverse=False):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            image_id, filename = line.strip().split()[0], line.strip().split()[1]
            if not reverse:
                mapping[filename] = int(image_id)-1
            else:
                mapping[int(image_id)-1] = filename

    return mapping

def load_concept_names(concepts_path):
    concept_names = {}
    with open(concepts_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                concept_id = int(parts[0])
                concept_name = parts[1]
                concept_names[concept_id] = concept_name

    return concept_names

def find_class_imbalance(concept_labels):
    _, num_concepts = concept_labels.shape
    concept_ratios = []
    for i in range(num_concepts):
        attribute_column = concept_labels[:, i]

        # Count occurrences of 0 and 1
        counts = np.bincount(attribute_column.astype(int), minlength=2)
        num_neg = counts[0]
        num_pos = counts[1]

        # Calculate ratio (handle division by zero)
        ratio_neg_pos = num_neg / num_pos if num_pos > 0 else float('inf') # Negatives per Positive

        concept_ratios.append(ratio_neg_pos)

    # concept_ratios_tensor = torch.tensor(concept_ratios, device=device, dtype=torch.float)
    return concept_ratios


def get_paths():
    """Get all paths needed for preprocessing."""
    dir_images = os.path.join(PROJECT_ROOT, 'images', 'Derm7pt')
    dir_data = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt')

    return {
        'dir_images': dir_images,
        'dir_data': dir_data,
        'meta_csv': os.path.join(dir_data, 'meta.csv'),
        'train_idx': os.path.join(dir_data, 'train_indexes.csv'),
        'val_idx': os.path.join(dir_data, 'valid_indexes.csv'),
        'test_idx': os.path.join(dir_data, 'test_indexes.csv'),
        'labels_file': os.path.join(dir_data, 'image_class_labels.txt'),
        'classes_path': os.path.join(dir_data, 'class_map.txt'),
        'mapping_file': os.path.join(dir_data, 'image_names.txt')
    }

def load_Derm_dataset(paths):
    """Load and prepare the dataset handler."""
    metadata_df = pd.read_csv(paths['meta_csv'])

    train_indexes = list(pd.read_csv(paths['train_idx'])['indexes'])
    valid_indexes = list(pd.read_csv(paths['val_idx'])['indexes'])
    test_indexes = list(pd.read_csv(paths['test_idx'])['indexes'])

    return Derm7PtDataset(
        dir_images=paths['dir_images'],
        metadata_df=metadata_df,
        train_indexes=train_indexes,
        valid_indexes=valid_indexes,
        test_indexes=test_indexes
    )