import os
from torch.utils.data import DataLoader, random_split

from src.config import RIVAL10_CONFIG, PROJECT_ROOT
from src.concept_dataset import ImageConceptDataset
from src.preprocessing import compute_class_level_concepts, apply_class_concepts_to_instances, select_common_concepts
from src.preprocessing.RIVAL10.data_encoding import get_concept_matrix, get_filename_mapping, one_hot_encode_labels
from src.preprocessing.RIVAL10.image_processing import load_and_transform_images
from src.utils.helpers import vprint


def preprocessing_rival10(training=False, class_concepts=False, verbose=False):
    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'train', 'ordinary')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'test', 'ordinary')

    TRAIN_IMG_PATH = os.path.join(PROJECT_ROOT, 'images', 'RIVAL10', 'train', 'images')
    TEST_IMG_PATH = os.path.join(PROJECT_ROOT, 'images', 'RIVAL10', 'test', 'images')

    # get filename to index mapping
    train_mapping = get_filename_mapping(TRAIN_IMG_PATH)
    test_mapping = get_filename_mapping(TEST_IMG_PATH)

    # get concept matrices
    train_concept_matrix = get_concept_matrix(train_mapping, TRAIN_PATH)
    test_concept_matrix = get_concept_matrix(test_mapping, TEST_PATH)

    vprint(f"Found {train_concept_matrix.shape[0] + test_concept_matrix.shape[0]} unique images.", verbose)
    vprint(f"Found {train_concept_matrix.shape[1]} unique concepts.", verbose)

    # get image labels
    labels_file = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'meta', 'wnid_to_class.json')
    super_labels_file = os.path.join(PROJECT_ROOT, 'data', 'RIVAL10', 'meta', 'label_mappings.json')

    train_img_labels = one_hot_encode_labels(labels_file, super_labels_file, train_mapping)
    test_img_labels = one_hot_encode_labels(labels_file, super_labels_file, test_mapping)

    vprint(f"Generated one-hot training matrix with shape: {train_img_labels.shape}", verbose)

    # get image tensors
    train_tensors, _ = load_and_transform_images(TRAIN_IMG_PATH, 224, training, train_mapping.values(), resnet=True, verbose=verbose)
    test_tensors, _ = load_and_transform_images(TEST_IMG_PATH, 224, training, test_mapping.values(), resnet=True, verbose=verbose)

    # concept processing
    class_level_concepts = compute_class_level_concepts(train_concept_matrix, None, train_img_labels)

    # apply class-level concepts to each instance
    if class_concepts:
        train_concept_matrix, test_concept_matrix = apply_class_concepts_to_instances(class_level_concepts, RIVAL10_CONFIG, train_img_labels, train_concept_matrix, test_img_labels, test_concept_matrix)

    common_concept_indices = select_common_concepts(class_level_concepts, min_class_count=0, CUB=False)
    train_concept_matrix = train_concept_matrix[:, common_concept_indices]
    test_concept_matrix = test_concept_matrix[:, common_concept_indices]

    # CREATE TRAIN AND TEST DATASET
    train_dataset = ImageConceptDataset(
        image_tensors=train_tensors,
        concept_labels=train_concept_matrix,
        image_labels=train_img_labels
    )

    test_dataset = ImageConceptDataset(
        image_tensors=test_tensors,
        concept_labels=test_concept_matrix,
        image_labels=test_img_labels
    )

    # Split train dataset into train and validation (80%/20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    vprint(f"Split train dataset: {train_size} training samples, {val_size} validation samples", verbose)

    # CREATE DATALOADERS FROM DATASETS
    batch_size = 64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=training, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return train_concept_matrix, train_loader, val_loader, test_loader