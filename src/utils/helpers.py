import os
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from derm7pt.dataset import Derm7PtDatasetGroupInfrequent


def vprint(message, is_verbose):
    if is_verbose:
        print(message)

def get_filename_to_id_mapping(filepath, reverse=False):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            image_id, filename = line.strip().split()[0], line.strip().split()[1]
            if not reverse:
                mapping[filename] = int(image_id)-1
            else:
                mapping[int(image_id)-1] = filename

    return mapping

def load_concept_names(concepts_path):
    concept_names = {}
    with open(concepts_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                concept_id = int(parts[0])
                concept_name = parts[1]
                concept_names[concept_id] = concept_name

    return concept_names

def find_class_imbalance(concept_labels):
    _, num_concepts = concept_labels.shape
    concept_ratios = []
    for i in range(num_concepts):
        attribute_column = concept_labels[:, i]

        # Count occurrences of 0 and 1
        counts = np.bincount(attribute_column.astype(int), minlength=2)
        num_neg = counts[0]
        num_pos = counts[1]

        # Calculate ratio (handle division by zero)
        ratio_neg_pos = num_neg / num_pos if num_pos > 0 else float('inf') # Negatives per Positive

        concept_ratios.append(ratio_neg_pos)

    # concept_ratios_tensor = torch.tensor(concept_ratios, device=device, dtype=torch.float)
    return concept_ratios


def get_paths():
    """Get all paths needed for preprocessing."""
    dir_images = os.path.join(PROJECT_ROOT, 'images', 'Derm7pt')
    dir_data = os.path.join(PROJECT_ROOT, 'data', 'Derm7pt')

    return {
        'dir_images': dir_images,
        'dir_data': dir_data,
        'meta_csv': os.path.join(dir_data, 'meta.csv'),
        'train_idx': os.path.join(dir_data, 'train_indexes.csv'),
        'val_idx': os.path.join(dir_data, 'valid_indexes.csv'),
        'test_idx': os.path.join(dir_data, 'test_indexes.csv'),
        'labels_file': os.path.join(dir_data, 'image_class_labels.txt'),
        'classes_path': os.path.join(dir_data, 'class_map.txt'),
        'mapping_file': os.path.join(dir_data, 'image_names.txt')
    }

def load_Derm_dataset(paths):
    """Load and prepare the dataset handler."""
    metadata_df = pd.read_csv(paths['meta_csv'])

    train_indexes = list(pd.read_csv(paths['train_idx'])['indexes'])
    valid_indexes = list(pd.read_csv(paths['val_idx'])['indexes'])
    test_indexes = list(pd.read_csv(paths['test_idx'])['indexes'])

    return Derm7PtDatasetGroupInfrequent(
        dir_images=paths['dir_images'],
        metadata_df=metadata_df,
        train_indexes=train_indexes,
        valid_indexes=valid_indexes,
        test_indexes=test_indexes
    )

def plot_explanation(data, binary_data, labels, total_dist):
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Colors
    binary_colors = ['#FF7F7F' if v > 0 else '#FFD17F' for v in data]
    bar_colors = ['#FF7F7F' if v > 0 else '#7FBF7F' for v in data]

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(data))

    # Plot bottom (binary)
    plt.bar(x, binary_data, edgecolor='black', color=binary_colors, linewidth=1, width=0.8)

    # Plot top (actual data)
    plt.bar(x, data + binary_data, edgecolor='black', color=bar_colors, linewidth=1, width=0.8)

    for i, (b_val, f_val) in enumerate(zip(binary_data, data)):
        # Label for binary part
        if b_val > 0:
            plt.text(i, (b_val + f_val) / 2, f'{np.abs(b_val+f_val):.2f}', ha='center', va='center', fontweight='bold', fontsize=14)
        # Label for main data part
        if np.abs(f_val) > 0.05:
            plt.text(i, b_val + f_val / 2, f'{np.abs(f_val):.2f}', ha='center', va='center', fontweight='bold', fontsize=14)


    plt.axvline(np.sum(data<0)-.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel('Concept', fontsize=14, fontweight='bold')
    plt.ylabel('Score/Distance', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(len(data)), labels)

    y_ticks = plt.yticks()[0]  # Get current tick positions
    new_y_ticks = y_ticks[:-1]
    plt.yticks(new_y_ticks, [f'{yt:.1f}' for yt in new_y_ticks], fontsize=14)

    plt.grid(axis='y', linestyle='-', alpha=0.3)

    plt.text(np.sum(data<0)/2-0.5, 1.1, 'Should Be Present',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))

    plt.text(np.sum(data<0) + np.sum(data>=0) / 2 - 0.5, 1.1, 'Should Not Be Present',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))

    legend_elements = [
        Patch(facecolor='white', edgecolor='white', label=f'Total Distance = {total_dist:.2f}'),
        Patch(facecolor='#7FBF7F', edgecolor='black', label='Correct Activation'),
        Patch(facecolor='#FFD17F', edgecolor='black', label='Missing Activation'),
        Patch(facecolor='#FF7F7F', edgecolor='black', label='Erroneous Activation')
    ]

    plt.legend(handles=legend_elements, loc='center right', prop={'family': 'sans-serif', 'weight': 'bold', 'size': 14})
    plt.tight_layout()
    plt.show()