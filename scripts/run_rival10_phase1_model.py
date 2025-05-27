import os
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp

from src.config import PROJECT_ROOT, RIVAL10_CONFIG
from src.utils import *
from src.training import run_epoch_x_to_c
from src.preprocessing.RIVAL10 import preprocessing_rival10

def main():
    N_TRIMMED_CONCEPTS = RIVAL10_CONFIG['N_TRIMMED_CONCEPTS']

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(42)
    concept_labels, train_loader, val_loader, test_loader = preprocessing_rival10(training=False, class_concepts=True, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(f"Using device: {device}")

    use_weighted_loss = True # Set to False for simple unweighted loss

    if use_weighted_loss:
        concept_weights = find_class_imbalance(concept_labels)
        attr_criterion = [nn.BCEWithLogitsLoss(weight=torch.tensor([ratio], device=device, dtype=torch.float))
                        for ratio in concept_weights]
    else:
        attr_criterion = [nn.BCEWithLogitsLoss() for _ in range(N_TRIMMED_CONCEPTS)]

    # Ensure output directory exists
    output_dir = os.path.join(PROJECT_ROOT, 'output', 'RIVAL10')
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, 'notebook', 'RIVAL10', 'x_to_c_best_model.pth')

    model = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Successfully loaded model from {model_path}")

    with torch.no_grad():
        shuffled_concept_labels = []
        shuffled_img_labels = []

        # Iterate through all batches
        for batch in train_loader:
            _, concept_labels, image_labels, _ = batch
            # Append batch labels to our list
            shuffled_concept_labels.append(concept_labels)
            shuffled_img_labels.append(image_labels)

        # Concatenate all batches into a single tensor
        shuffled_concept_labels = torch.cat(shuffled_concept_labels, dim=0)
        shuffled_img_labels = torch.cat(shuffled_img_labels, dim=0)

        test_loss, test_acc, outputs = run_epoch_x_to_c(
            model, train_loader, attr_criterion, optimizer=None, n_concepts=N_TRIMMED_CONCEPTS, device=device,
            return_outputs='sigmoid', verbose=True
        )

    # print(f"Shuffled labels shape: {shuffled_img_labels.shape}")
    np.save(os.path.join(PROJECT_ROOT, 'output', 'RIVAL10', 'C_train.npy'), shuffled_concept_labels)
    np.save(os.path.join(PROJECT_ROOT, 'output', 'RIVAL10', 'Y_train.npy'), shuffled_img_labels)
    print(f'Best Model Summary   | Loss: {test_loss:.4f} | Acc: {test_acc:.3f}')

    output_array = get_outputs_as_array(outputs, N_TRIMMED_CONCEPTS)
    print(f"Final shape: {output_array.shape}")

    np.save(os.path.join(output_dir, 'C_hat_sigmoid_train.npy'), output_array)

def get_outputs_as_array(outputs, n_concepts):
    # Initialize an empty list to collect batches
    batch_results = []

    for i in range(len(outputs)):
        batch_size = outputs[i].shape[0]

        # Create a batch matrix with n_concepts number of columns
        batch_matrix = np.zeros((batch_size, n_concepts))

        for instance_idx in range(batch_size):
            # Extract, convert, and flatten data for the current concept
            instance_data = outputs[i][instance_idx].detach().cpu().numpy().flatten()
            batch_matrix[instance_idx, :] = instance_data

        # Add this consistently shaped batch matrix to our collection
        batch_results.append(batch_matrix)

    return np.vstack(batch_results)

if __name__ == '__main__':
    # This is crucial for multiprocessing with PyTorch DataLoader
    mp.set_start_method('spawn', force=True)
    main()