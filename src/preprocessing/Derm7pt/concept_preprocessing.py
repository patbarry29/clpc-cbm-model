import os
import numpy as np
import pandas as pd
from config import PROJECT_ROOT
from src.utils import vprint

def encode_image_concepts(dataset, verbose=False):
    all_data = dataset.get_labels(data_type='all', one_hot=True)

    concepts_matrix = None
    concept_meanings = []
    column_index = 0

    for tag in dataset.tags.abbrevs:
        if tag == 'DIAG':
            continue

        feature_set = all_data[tag]

        if concepts_matrix is None:
            concepts_matrix = feature_set
        else:
            # Concatenate horizontally (along axis 1)
            concepts_matrix = np.hstack((concepts_matrix, feature_set))

        tag_definitions = dataset.get_label_by_abbrev(tag)

        # Store the meaning of each column for this tag
        num_concepts = feature_set.shape[1]
        for i in range(num_concepts):
            name = tag_definitions.names[i]

            concept_meanings.append((tag, name))

    concept_meanings = np.asarray(concept_meanings, dtype="object")

    image_names_path = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt', 'image_names.txt')
    flattened_df = pd.read_csv(image_names_path, sep=' ', header=None, names=['img_id', 'img_path', 'img_type', 'case_id'])

    all_concepts = []

    for _, row in flattened_df.iterrows():
        case_id = row['case_id']
        case_concepts = concepts_matrix[case_id]
        all_concepts.append(case_concepts)

    all_concepts = np.array(all_concepts)
    all_concepts.shape

    vprint(f"Total number of concept columns: {all_concepts.shape[1]}", verbose)

    return all_concepts