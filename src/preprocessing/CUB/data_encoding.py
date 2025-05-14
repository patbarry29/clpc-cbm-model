import numpy as np
import pandas as pd

from src.utils.helpers import load_concept_names, vprint

def _parse_file(concept_labels_file):
    data = []
    with open(concept_labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # We only need the first 3 columns (image_id, concept_id, is_present)
            image_id = int(parts[0])
            concept_id = int(parts[1])
            is_present = int(parts[2])
            uncertainty = int(parts[3])
            data.append([image_id, concept_id, is_present, uncertainty])

    return data

def encode_image_concepts(concept_labels_file, verbose=False):
    # 1. get image_id, concept_id and is_present values from file
    data = _parse_file(concept_labels_file)

    # 2. Create a DataFrame from the parsed data
    concept_df = pd.DataFrame(data, columns=['image_id', 'concept_id', 'is_present', 'uncertainty'])

    # 3. get the number of unique images and concepts
    unique_images = concept_df['image_id'].unique()
    num_images = len(unique_images)

    # -- find the max concept_id to determine matrix dimensions
    max_concept_id = concept_df['concept_id'].max()

    vprint(f"Found {num_images} unique images.", verbose)
    vprint(f"Found {max_concept_id} unique concepts.", verbose)

    # 4. create concepts matrix initialized with zeros
    concept_matrix = np.zeros((num_images, max_concept_id), dtype=int)
    uncertainty_matrix = np.zeros((num_images, max_concept_id), dtype=int)

    # 5. populate matrix (vectorised)
    img_ids = concept_df['image_id'].values - 1
    concept_ids = concept_df['concept_id'].values - 1
    concept_matrix[img_ids, concept_ids] = concept_df['is_present'].values
    uncertainty_matrix[img_ids, concept_ids] = concept_df['uncertainty'].values

    vprint(f"Generated concept matrix with shape: {concept_matrix.shape}", verbose)

    return concept_matrix, uncertainty_matrix


def get_concepts(concept_vector, concepts_path):
    concept_names = load_concept_names(concepts_path)

    true_concept_indices = np.where(concept_vector == 1)[0]

    true_concept_ids = true_concept_indices + 1

    active_concepts = [concept_names[concept_id] for concept_id in true_concept_ids if concept_id in concept_names]

    return active_concepts