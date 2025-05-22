import numpy as np
import pandas as pd

from src.utils.helpers import vprint


def one_hot_encode_labels(dataset, image_names_path, verbose=False):
    all_labels = dataset.get_labels(data_type='all', one_hot=True)
    mapping_data = pd.read_csv(image_names_path, sep=' ', header=None, names=['img_id', 'img_path', 'img_type', 'case_id'])

    labels_matrix = []

    for _, row in mapping_data.iterrows():
        case_id = row['case_id']
        label_one_hot = all_labels['DIAG'][case_id]

        labels_matrix.append(label_one_hot)

    labels_matrix = np.asarray(labels_matrix)
    vprint(f"Total number of label columns: {labels_matrix.shape[1]}", verbose)
    vprint(f"Found {labels_matrix.shape[0]} instances.", verbose)
    vprint(f"Created matrix of shape: {labels_matrix.shape}", verbose)

    # Remove columns 0 and 4
    # labels_matrix = np.delete(labels_matrix, [0, 3, 4], axis=1)
    # vprint(f"Matrix shape after removing columns 0 and 4: {labels_matrix.shape}", verbose)

    return labels_matrix