import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class A2_DeepACTIF_Forward_OLD:
    def __init__(self, model, features=None):
        self.model = model
        self.selected_features = features if features is not None else []

    def forward_pass_with_importance(self, inputs):
        """
        Passes inputs through each layer, collects activations, and interpolates importance back to input features.
        """
        self.model.eval()
        activations = inputs
        accumulated_importance = torch.zeros((inputs.shape[0], len(self.selected_features)), device=inputs.device)

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.LSTM):
                activations, _ = layer(activations)
                layer_importance = activations.sum(dim=1)  # (batch, features)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)
                layer_importance = activations.mean(dim=0)  # (features,)
            else:
                continue  # Skip non-trainable layers

            # Ensure layer_importance is at least 3D before interpolation
            if layer_importance.dim() == 2:  # Shape (batch, features)
                layer_importance = layer_importance.unsqueeze(1)  # Convert to (batch, 1, features)
            elif layer_importance.dim() == 1:  # Shape (features,)
                layer_importance = layer_importance.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, features)

            print(f"Layer {name} -> Shape before interpolation: {layer_importance.shape}")

            # **Fix Interpolation**
            interpolated_importance = F.interpolate(
                layer_importance,  # Ensure it's (batch, 1, features)
                size=(len(self.selected_features),),  # Target feature count (must be a tuple)
                mode="linear",
                align_corners=False
            ).squeeze(1)  # Remove the extra dimension after interpolation -> (batch, features)

            # 🔍 Debugging shapes before addition
            print(f"Accumulated importance shape: {accumulated_importance.shape}")
            print(f"Interpolated importance shape: {interpolated_importance.shape}")

            # Ensure shape compatibility before addition
            min_dim = min(accumulated_importance.shape[0], interpolated_importance.shape[0])
            accumulated_importance = accumulated_importance[:min_dim, :]
            interpolated_importance = interpolated_importance[:min_dim, :]

            # Accumulate importance
            accumulated_importance += interpolated_importance

        return accumulated_importance.mean(dim=0)

    def analyze(self, dataloader, device):
        """
        Computes feature importance over a dataset.
        """
        all_importances = []

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Extract inputs if (inputs, labels) tuple

            batch = batch.to(device, dtype=torch.float32)
            feature_importance = self.forward_pass_with_importance(batch)

            if feature_importance is None or torch.isnan(feature_importance).any():
                print(f"⚠️ Skipping batch {batch_idx}, feature importance contains NaN!")
                continue

            all_importances.append(feature_importance)

        if not all_importances:
            raise ValueError("❌ No valid feature importances calculated!")

        # Convert tensors to NumPy
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp for imp in
                           all_importances]
        all_attributions = np.vstack(all_importances)

        # **Debugging Check for NaNs**
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in all_attributions!")
            print("NaN Locations:", np.where(np.isnan(all_attributions)))
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)

        print(f"✅ Final shape of all_attributions: {all_attributions.shape}")

        # Apply ACTIF variant for aggregation
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)

        # **Handle NaNs in Importance Calculation**
        if np.isnan(importance).any():
            print("⚠️ Replacing NaN values in final feature importance")
            importance = np.nan_to_num(importance, nan=0.0)

        print(f"✅ Calculated importance shape: {importance.shape}")

        # # Ensure correct shape
        # if importance.shape[0] != len(self.selected_features):
        #     raise ValueError(
        #         f"❌ ACTIF method returned {importance.shape[0]} scores, but expected {len(self.selected_features)} features."
        #     )

        # # Prepare final results
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]

        return results

        # #  🔹 Return results **per sample**
        # results = [
        #     [{'feature': self.selected_features[i], 'attribution': importance[sample_idx, i]}
        #     for i in range(len(self.selected_features))]
        #     for sample_idx in range(importance.shape[0])
        # ]

        return results  # List of dicts per sample

    def calculate_actif_inverted_weighted_mean(self, activation):
        """
        Calculate the importance by weighting high mean activations and low variability (stddev).

        Args:
            activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

        Returns:
            adjusted_importance (np.ndarray): Adjusted importance where low variability is rewarded.
        """
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        # Normalize mean and invert normalized stddev
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation))
        inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (
                np.max(std_activation) - np.min(std_activation))

        # Calculate importance as a combination of mean and inverse stddev
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance
