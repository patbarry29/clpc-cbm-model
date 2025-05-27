from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd

from .concept_preprocessing import encode_image_concepts
from .dataset_utils import export_image_props_to_text, filter_concepts_labels
from .label_encoding import one_hot_encode_labels
from .split_data import split_data_by_indices
from src.concept_dataset import ImageConceptDataset
from src.config import DERM7PT_CONFIG, PROJECT_ROOT
from src.preprocessing import *
from src.utils import get_paths, load_Derm_dataset


def preprocessing_Derm7pt(training=False, class_concepts=True, verbose=False):
    paths = get_paths()
    dataset_handler = load_Derm_dataset(paths)

    # Ensure text files exist
    if not os.path.exists(paths['labels_file']):
        export_image_props_to_text(dataset_handler.df)

    # Get labels and concepts
    image_labels = one_hot_encode_labels(dataset_handler, paths['mapping_file'], verbose=verbose)
    concepts_matrix = encode_image_concepts(dataset_handler, paths['mapping_file'], verbose=verbose)

    # Load and transform images
    image_tensors, image_paths = load_and_transform_images(paths['dir_images'], paths['mapping_file'], resol=224, use_training_transforms=training, batch_size=64, resnet=True, verbose=verbose)

    # Filter if needed
    if image_labels.shape[0] != len(image_tensors):
        filtered_image_labels, filtered_concepts_matrix = filter_concepts_labels(
            paths['mapping_file'], image_tensors, image_paths, image_labels, concepts_matrix
        )
    else:
        filtered_image_labels, filtered_concepts_matrix = image_labels, concepts_matrix

    if verbose:
        print("Labels shape:", filtered_image_labels.shape)
        print("Concepts shape:", filtered_concepts_matrix.shape)
        print("Image tensors length:", len(image_tensors))

    # Split data into train, validation and test sets if requested
    tensors_dict, concepts_dict, labels_dict = split_data_by_indices(
        image_tensors, image_paths, filtered_concepts_matrix, filtered_image_labels,
        paths, verbose=verbose
    )

    train_concept_labels = concepts_dict['train']
    val_concept_labels = concepts_dict['val']
    test_concept_labels = concepts_dict['test']

    train_img_labels = labels_dict['train']
    val_img_labels = labels_dict['val']
    test_img_labels = labels_dict['test']

    train_tensors = tensors_dict['train']
    val_tensors = tensors_dict['val']
    test_tensors = tensors_dict['test']

    # concept processing
    class_level_concepts = compute_class_level_concepts(train_concept_labels, None, train_img_labels)

    # apply class-level concepts to each instance
    if class_concepts:
        train_concept_labels, val_concept_labels, test_concept_labels = apply_class_concepts_to_instances(
            class_level_concepts, DERM7PT_CONFIG, train_img_labels, train_concept_labels,
            test_img_labels, test_concept_labels, val_img_labels, val_concept_labels)

    common_concept_indices = select_common_concepts(class_level_concepts, min_class_count=0, CUB=False)
    train_concept_labels = train_concept_labels[:, common_concept_indices]
    val_concept_labels = val_concept_labels[:, common_concept_indices]
    test_concept_labels = test_concept_labels[:, common_concept_indices]

    # CREATE TRAIN AND TEST DATASET
    train_dataset = ImageConceptDataset(
        image_tensors=train_tensors,
        concept_labels=train_concept_labels,
        image_labels=train_img_labels
    )

    val_dataset = ImageConceptDataset(
        image_tensors=val_tensors,
        concept_labels=val_concept_labels,
        image_labels=val_img_labels
    )

    test_dataset = ImageConceptDataset(
        image_tensors=test_tensors,
        concept_labels=test_concept_labels,
        image_labels=test_img_labels
    )

    # CREATE DATALOADERS FROM DATASETS
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return filtered_concepts_matrix, train_loader, val_loader, test_loader


if __name__ == '__main__':
    concepts_matrix, train_loader, test_loader = preprocessing_Derm7pt(class_concepts=True, verbose=True)
