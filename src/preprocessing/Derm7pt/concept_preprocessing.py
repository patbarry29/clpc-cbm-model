import os
import numpy as np
import pandas as pd
from src.config import PROJECT_ROOT
from src.utils import vprint

def encode_image_concepts(dataset, image_names_path, verbose=False):
    all_labels = dataset.get_labels(data_type='all', one_hot=True)
    mapping_data = pd.read_csv(image_names_path, sep=' ', header=None, names=['img_id', 'img_path', 'img_type', 'case_id'])

    concepts_matrix = []

    for _, row in mapping_data.iterrows():
        case_id = row['case_id']
        instance_concepts = None
        for concept, vals in all_labels.items():
            if concept == 'DIAG':
                continue

            concept_one_hot = vals[case_id]

            if instance_concepts is None:
                instance_concepts = concept_one_hot
            else:
                # Concatenate horizontally (along axis 1)
                instance_concepts = np.hstack((instance_concepts, concept_one_hot))

        concepts_matrix.append(instance_concepts)

    concepts_matrix = np.asarray(concepts_matrix)
    vprint(f"Total number of concept columns: {concepts_matrix.shape[1]}", verbose)

    return concepts_matrix
