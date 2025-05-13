from PIL import Image
import os

import torchvision.transforms as transforms
import math

from src.utils import *

from tqdm import tqdm


def resize_images(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    for dir in os.listdir(input_dir):
        curr_dir = os.path.join(input_dir, dir)

        if not os.path.isdir(curr_dir):
            continue

        os.makedirs(os.path.join(output_dir, dir), exist_ok=True)

        for filename in os.listdir(curr_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(curr_dir, filename)
                with Image.open(img_path) as img:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img.save(os.path.join(output_dir, dir, filename))


def _get_transform_pipeline(use_training_transforms, resol, resnet=False):
    # resized_resol - resizes to slightly larger to ensure image is large enough before cropping to resol
    resized_resol = int(resol * 256/224) # 299 * 256/224 = 341.7
    if resnet:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else: # shifts from [0.0, 1.0] -> [-0.25, 0.25]
        mean = [0.5,0.5,0.5]
        std = [2,2,2]
    if use_training_transforms:
        # print("Using TRAINING transformations:")
        return transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
            ])
    # Use LANCZOS resampling for better quality
    # print("Using VALIDATION/TEST transformations:")
    return transforms.Compose([
        transforms.Resize(resized_resol, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resol),
        transforms.ToTensor(), # divide by 255, convert from [0,255] -> [0.0,1.0]
        transforms.Normalize(mean = mean, std = std)
        ])


def _get_all_filenames(input_dir):
    all_image_paths = []

    for subdir, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_image_paths.append(os.path.join(subdir, filename))

    return all_image_paths

def _get_filename_from_path(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    return os.path.join(parent_folder, filename)


def load_and_transform_images(input_dir, mapping_file, resol, use_training_transforms, batch_size = 64, resnet=False, verbose = False):
    """
    Returns:
        A tuple containing:
        - A list of all transformed image tensors.
        - A list of the corresponding original file paths for each tensor.
    """
    # Get the transformation pipeline
    transform_pipeline = _get_transform_pipeline(use_training_transforms, resol, resnet)

    # Get all image paths
    all_image_paths = _get_all_filenames(input_dir)

    # get correct sorting
    if mapping_file:
        filename_id_map = get_filename_to_id_mapping(mapping_file)
        # Create a case-insensitive lookup dictionary
        filename_id_map_lower = {k.lower(): v for k, v in filename_id_map.items()}
        # Sort using case-insensitive lookup
        all_image_paths.sort(key=lambda path: filename_id_map_lower.get(_get_filename_from_path(path).lower(), float('inf')))

    vprint(f"Found {len(all_image_paths)} images.", verbose)
    num_batches = math.ceil(len(all_image_paths) / batch_size)
    vprint(f"Processing in {num_batches} batches of size {batch_size} (for progress reporting)...", verbose)

    all_transformed_tensors = []
    all_processed_paths = []
    processed_count = 0

    for i in tqdm(range(num_batches), desc="Processing batches", disable=not verbose):
        batch_paths = all_image_paths[i * batch_size : (i + 1) * batch_size]

        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            # Apply transformations
            transformed_img_tensor = transform_pipeline(img)
            # Append the tensor and its path to the lists
            all_transformed_tensors.append(transformed_img_tensor)
            img_path = _get_filename_from_path(img_path)
            all_processed_paths.append(img_path)
            processed_count += 1

    vprint(f"\nFinished processing.", verbose)
    vprint(f"Successfully transformed: {processed_count} images.", verbose)

    # Return the list of all tensors and their paths
    return all_transformed_tensors, all_processed_paths
