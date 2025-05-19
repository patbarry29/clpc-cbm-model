import numpy as np
import pandas as pd
import os

def split_data_by_indices(image_tensors, image_paths, concepts_matrix, image_labels, paths, verbose=False):
    # Load the indices
    train_indices_df = pd.read_csv(paths['train_idx'])
    val_indices_df = pd.read_csv(paths['val_idx'])
    test_indices_df = pd.read_csv(paths['test_idx'])

    train_indices = list(train_indices_df['indexes'])
    val_indices = list(val_indices_df['indexes'])
    test_indices = list(test_indices_df['indexes'])

    # Load mapping file to get case numbers
    mapping_df = pd.read_csv(paths['mapping_file'], sep=' ', header=None,
                            names=['img_id', 'img_path', 'img_type', 'case_num'])

    # Create a mapping from image path to index in our arrays
    path_to_index = {path.upper(): i for i, path in enumerate(image_paths)}

    # Function to get indices for a split
    def get_split_indices(case_indices):
        indices = []
        for case_num in case_indices:
            # Find rows in mapping_df that match this case_num
            case_rows = mapping_df[mapping_df['case_num'] == case_num]

            for _, row in case_rows.iterrows():
                img_path = row['img_path'].upper()
                if img_path in path_to_index:
                    idx = path_to_index[img_path]
                    indices.append(idx)
        return indices

    # Get indices for each split
    train_img_indices = get_split_indices(train_indices)
    val_img_indices = get_split_indices(val_indices)
    test_img_indices = get_split_indices(test_indices)

    # Split the data
    train_tensors = [image_tensors[i] for i in train_img_indices]
    val_tensors = [image_tensors[i] for i in val_img_indices]
    test_tensors = [image_tensors[i] for i in test_img_indices]

    train_concepts = concepts_matrix[train_img_indices]
    val_concepts = concepts_matrix[val_img_indices]
    test_concepts = concepts_matrix[test_img_indices]

    train_labels = image_labels[train_img_indices]
    val_labels = image_labels[val_img_indices]
    test_labels = image_labels[test_img_indices]

    tensors_dict = {
        'train': train_tensors,
        'val': val_tensors,
        'test': test_tensors
    }

    concepts_dict = {
        'train': train_concepts,
        'val': val_concepts,
        'test': test_concepts
    }

    labels_dict = {
        'train': train_labels,
        'val': val_labels,
        'test': test_labels
    }

    return tensors_dict, concepts_dict, labels_dict
