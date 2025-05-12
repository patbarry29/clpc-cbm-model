import numpy as np
import pandas as pd

from src.utils.helpers import vprint


def one_hot_encode_labels(image_class_path, classes_path, verbose=False):
    # 1. determine the number of classes
    classes_df = pd.read_csv(classes_path, sep=' ', header=None, names=['class_id', 'class_name'])
    num_classes = len(classes_df)
    vprint(f"Found {num_classes} classes.", verbose)

    # 2. get image labels
    labels_df = pd.read_csv(image_class_path, sep=' ', header=None, names=['image_id', 'class_id'])
    num_images = len(labels_df)
    vprint(f"Found labels for {num_images} images.", verbose)

    # 3. initialise label matrix with zeros
    one_hot_matrix = np.zeros((num_images, num_classes), dtype=int)

    # 4. populate matrix
    class_ids = labels_df['class_id'].values - 1
    one_hot_matrix[np.arange(len(labels_df)), class_ids] = 1

    vprint(f"Generated one-hot matrix with shape: {one_hot_matrix.shape}", verbose)
    return one_hot_matrix