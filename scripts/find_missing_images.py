#!/usr/bin/env python
# Script to find images in the mapping file that don't exist in the actual directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processing import _get_all_filenames, _get_filename_from_path
from src.utils import get_filename_to_id_mapping
from src.config import PROJECT_ROOT

def find_missing_images(input_dir, mapping_file):
    """
    Find images that are in the mapping file but not in the actual directory.

    Args:
        input_dir: Directory containing the images
        mapping_file: File with the image name to ID mapping

    Returns:
        A list of image names that are in the mapping file but not in the directory
    """
    # Get all actual image paths
    all_image_paths = _get_all_filenames(input_dir)
    actual_image_names = [_get_filename_from_path(path) for path in all_image_paths]

    # Create case insensitive lookup dictionaries
    actual_image_dict = {name.lower(): name for name in actual_image_names}

    # Get all image names from the mapping file
    filename_id_map = get_filename_to_id_mapping(mapping_file)
    mapped_image_names = list(filename_id_map.keys())
    mapped_image_dict = {name.lower(): name for name in mapped_image_names}

    # Find differences using case-insensitive comparisons
    missing_images = []
    for mapped_name in mapped_image_names:
        if mapped_name.lower() not in actual_image_dict:
            missing_images.append(mapped_name)

    extra_images = []
    for actual_name in actual_image_names:
        if actual_name.lower() not in mapped_image_dict:
            extra_images.append(actual_name)

    # Find case mismatches (same name but different case)
    case_mismatches = []
    for mapped_name in mapped_image_names:
        mapped_lower = mapped_name.lower()
        if mapped_lower in actual_image_dict and mapped_name != actual_image_dict[mapped_lower]:
            case_mismatches.append((mapped_name, actual_image_dict[mapped_lower]))

    # Print statistics
    print(f"Images in mapping file: {len(mapped_image_names)}")
    print(f"Images in directory: {len(actual_image_names)}")

    return missing_images, extra_images, case_mismatches

if __name__ == "__main__":
    # You can adjust these paths as needed
    data_dir = os.path.join(PROJECT_ROOT, "images", "Derm7pt")
    mapping_file = os.path.join(PROJECT_ROOT, "data", "Derm7pt", "image_names.txt")

    missing_images, extra_images, case_mismatches = find_missing_images(data_dir, mapping_file)

    print("\nImages in mapping file but missing from directory:")
    for img in sorted(missing_images):
        print(f"  - {img}")
    print(f"Total missing: {len(missing_images)}")

    print("\nImages in directory but not in mapping file:")
    for img in sorted(extra_images):
        print(f"  - {img}")
    print(f"Total extra: {len(extra_images)}")

    print("\nCase mismatches (mapping file vs. directory):")
    for mapped, actual in sorted(case_mismatches):
        print(f"  - {mapped} ≠ {actual}")
    print(f"Total case mismatches: {len(case_mismatches)}")

    # Suggest fixes for case mismatches
    if case_mismatches:
        print("\nSuggested fix for case mismatches:")
        print("1. Update the mapping file to match actual filenames:")
        for mapped, actual in sorted(case_mismatches):
            print(f"   Replace '{mapped}' with '{actual}'")

        print("\n2. Or rename the files to match the mapping:")
        for mapped, actual in sorted(case_mismatches):
            print(f"   mv \"{os.path.join(data_dir, actual)}\" \"{os.path.join(data_dir, mapped)}\"")
