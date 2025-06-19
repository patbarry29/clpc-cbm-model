import numpy as np
import torch
import torch.nn as nn

from src.utils.helpers import plot_explanation

class PrototypeClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.rand(num_classes, num_features))  # initialize the prototype matrix P

    def forward(self, x):
        # x: (batch_size, num_features)
        # L1distance：|x_i - M_m|_1
        # (batch_size, num_classes, num_features)
        dist = torch.abs(x.unsqueeze(1) - torch.sigmoid(self.prototypes))
        dist = dist.sum(dim=2)
        return dist  # (batch_size, num_classes)

    def binary_regularization(self):
        sigmoid_protos = torch.sigmoid(self.prototypes)
        return (sigmoid_protos * (1 - sigmoid_protos)).mean()

    def sparsity_regularization(self):
        return torch.sum(torch.sigmoid(self.prototypes))

    def predict(self, x):
        with torch.no_grad():
            Prototypes = torch.sigmoid(self.prototypes)
            Prototypes[Prototypes>=0.5] = 1
            Prototypes[Prototypes<0.5]= 0
            dists = torch.abs(x.unsqueeze(1) - Prototypes)
            dists = dists.sum(dim=2)
            predictions = dists.argmin(dim=1)
        return predictions

    def get_sigmoid_prototypes(self):
        return torch.sigmoid(self.prototypes)

    def get_binary_prototypes(self):
        Prototypes = torch.sigmoid(self.prototypes)
        Prototypes[Prototypes>=0.5] = 1
        Prototypes[Prototypes<0.5]= 0
        return Prototypes

    def concept_wise_dist(self, x):
        with torch.no_grad():
            Prototypes = torch.sigmoid(self.prototypes)
            Prototypes[Prototypes>=0.5] = 1
            Prototypes[Prototypes<0.5]= 0
            dists = x.unsqueeze(1) - Prototypes
            # predictions = self.predict(x)
            # dists = dists[torch.arange(x.shape[0]), predictions,:]
        return dists

    def threshold(self, val_x, val_y, percentile=0.98):
        self.eval()
        with torch.no_grad():
            dists = self(val_x)
            min_dists = dists.min(dim=1).values.cpu().numpy()

            # y_val is one-hot or not ?
            if val_y.ndim > 1 and val_y.shape[1] > 1:
                real_labels = val_y.argmax(dim=1).cpu()
            else:
                real_labels = val_y

            pred_labels = self.predict(val_x).cpu()
            matching_array = (pred_labels == real_labels).numpy()

            correct_min_dists = min_dists[matching_array]
            correct_min_dists.sort()

            self.computed_threshold_ = np.percentile(correct_min_dists, percentile * 100)
            print(f"Threshold computed: {self.computed_threshold_:.4f} using {len(correct_min_dists)} correctly classified validation samples at {percentile*100:.1f}th percentile.")
        return self.computed_threshold_

    def outlier_predict(self, x):
        self.eval()
        with torch.no_grad():
            dists_x = self(x)
            min_dists_x = dists_x.min(dim=1).values
            predictions = self.predict(x)

            # Identify inliers
            is_inlier = min_dists_x.cpu() <= self.computed_threshold_

            # Return predictions, but mark outliers with a -1
            conformal_predictions = predictions.clone()
            conformal_predictions[~is_inlier.to(predictions.device)] = -1

        return conformal_predictions, is_inlier

    def conformal_predict(self, x):
        pass


    def get_uncertainties(self, x, y_true):
        pred = self.predict(x.unsqueeze(0))
        print("Classification Correct:", (y_true==pred).item())
        return x - self.get_binary_prototypes()[pred]

    def explanation(self, x, y_true):
        concept_uncertainties = self.get_uncertainties(x, y_true)

        uncertainties = concept_uncertainties.cpu().detach().numpy()[0]
        top_indices = np.argsort(np.abs(uncertainties))[::-1][:10]
        top_uncertainties = uncertainties[top_indices]  # Work on top 10 values

        # Masks
        mask_neg = top_uncertainties <= 0
        mask_pos = top_uncertainties > 0

        # Sort within top 10
        neg_indices = np.argsort(-top_uncertainties[mask_neg])  # ascending
        pos_indices = np.argsort(-top_uncertainties[mask_pos])  # descending

        # Map sorted positions back to indices in top_indices, then to original
        neg_sorted_indices = top_indices[mask_neg][neg_indices]
        pos_sorted_indices = top_indices[mask_pos][pos_indices]

        # Combine indices and values
        final_sorted_indices = np.concatenate([neg_sorted_indices, pos_sorted_indices])
        final_sorted_data = uncertainties[final_sorted_indices]

        # Binary overlay
        binary_data = final_sorted_data.copy()
        binary_data[final_sorted_data <= 0] = 1
        binary_data[final_sorted_data > 0] = 0

        # Labels
        sorted_labels = [fr'$c_{{{idx+1}}}$' for idx in final_sorted_indices]

        # Distance
        total_distance = np.sum(np.abs(uncertainties))

        plot_explanation(final_sorted_data, np.sort(binary_data)[::-1], sorted_labels, total_distance)

    def modify_prototypes(self, new_prototypes):
        self.prototypes.data = new_prototypes






# class PrototypeLearner(nn.Module):
#     def forward(self, C_hat, Y_true=None, lambda_bin=0.1, lambda_spars=0.01):
#         # Get continuous prototypes
#         prototypes = self.prototypes.weight
#         prototypes_sigmoid = torch.sigmoid(prototypes)  # Shape: [num_classes, num_concepts]

#         # Calculate absolute difference between concepts and prototypes
#         concept_distances = torch.abs(C_hat.unsqueeze(1) - prototypes_sigmoid)  # Shape: [batch_size, num_classes, num_concepts]
#         # Sum distances across concept dimension
#         label_distances = concept_distances.sum(dim=2)  # Shape: [batch_size, num_classes]
#         pred_label = label_distances.argmin(dim=1)  # Shape: [batch_size]

#         # Classification loss - using the distances for labeled classes
#         loss_class = torch.mean(torch.sum(label_distances * Y_true, dim=1))

#         # Binarization loss - encourages prototypes to be binary (0 or 1)
#         loss_bin = torch.mean(prototypes_sigmoid * (1 - prototypes_sigmoid))

#         # Sparsity loss - encourages fewer active concepts
#         loss_spars = torch.mean(torch.abs(prototypes_sigmoid))

#         # Combine losses
#         total_loss = loss_class + (lambda_bin * loss_bin) + (lambda_spars * loss_spars)

#         return pred_label, total_loss

#     def get_binary_prototypes(self):
#         with torch.no_grad():
#             binary_prototypes = (torch.sigmoid(self.prototypes.weight) > 0.5).float()
#         return binary_prototypes

#     def get_sigmoid_prototypes(self):
#         return torch.sigmoid(self.prototypes.weight)