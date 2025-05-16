import os
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

def export_image_props_to_text(dataset):
    all_images = []
    all_types = []
    case_ids = []

    for idx, row in dataset.iterrows():
        all_images.append(row['clinic'])
        all_types.append('clinic')
        case_ids.append(idx)

        all_images.append(row['derm'])
        all_types.append('derm')
        case_ids.append(idx)

    flattened_df = pd.DataFrame({
        'image_path': all_images,
        'image_type': all_types,
        'case_id': case_ids
    })

    # First get all unique class names
    unique_class_names = flattened_df['image_path'].apply(lambda x: x.split('/')[0].upper()).unique()
    # Create a mapping from class names to integers (starting from 1)
    class_to_int = {cls_name: idx+1 for idx, cls_name in enumerate(sorted(unique_class_names))}

    def extract_label(file_path):
        class_name = file_path.split('/')[0].upper()
        return class_to_int[class_name]

    flattened_df['image_label'] = flattened_df['image_path'].apply(extract_label)

    class_map_path = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt', 'class_map.txt')
    with open(class_map_path, 'w') as f:
        for cls_name, idx in class_to_int.items():
            f.write(f"{idx} {cls_name}\n")

    # all_concepts = []

    # for _, row in flattened_df.iterrows():
    #     case_id = row['case_id']
    #     case_concepts = concepts_matrix[case_id]
    #     all_concepts.append(case_concepts)

    # all_concepts_array = np.array(all_concepts)

    # flattened_df['concepts'] = all_concepts

    image_names_path = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt', 'image_names.txt')
    flattened_df[['image_path', 'image_type', 'case_id']].to_csv(image_names_path, sep=' ', index=True, header=False)

    image_labels_path = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt', 'image_class_labels.txt')
    flattened_df[['image_label']].to_csv(image_labels_path, sep=' ', index=True, header=False)



def filter_concepts_labels(mapping_file, image_tensors, image_paths, image_labels, concepts_matrix):
    processed_paths_map = {path.upper(): i for i, path in enumerate(image_paths)}

    with open(mapping_file, 'r') as f:
        lines = f.readlines()

    filtered_image_labels = np.zeros((len(image_tensors), image_labels.shape[1]), dtype=image_labels.dtype)
    filtered_concepts_matrix = np.zeros((len(image_tensors), concepts_matrix.shape[1]), dtype=concepts_matrix.dtype)

    skipped_count = 0

    for line in lines:
        parts = line.strip().split()
        original_idx = int(parts[0])
        file_path = parts[1]

        # Check if this file was processed (using case insensitive comparison)
        if file_path.upper() in processed_paths_map:
            if original_idx < len(concepts_matrix):
                new_idx = processed_paths_map[file_path.upper()]
                filtered_image_labels[new_idx] = image_labels[original_idx]
                filtered_concepts_matrix[new_idx] = concepts_matrix[original_idx]
            else:
                skipped_count += 1
                print(f"Warning: Index {original_idx} is out of bounds for concepts_matrix with size {len(concepts_matrix)}. Skipping.")

    if skipped_count > 0:
        print(f"Total indices skipped due to being out of bounds: {skipped_count}")

    return filtered_image_labels, filtered_concepts_matrix