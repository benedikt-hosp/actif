import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


class A4_DeepACTIF_SingleLayer:
    def __init__(self, model, selected_features, device):
        """
        Args:
            model (torch.nn.Module): Das zu analysierende Modell.
            selected_features (list): Liste der Feature-Namen (z. B. 34 Einträge).
            device (str): Device, z. B. "cuda", "mps" oder "cpu".
        """
        self.model = model
        self.hook_location = "lstm"
        self.selected_features = selected_features
        self.device = device
        self.model = model

    def compute(self, valid_loader):
        """
            deepactif calculation with different layer hooks based on version.
            Args:
                hook_location: Where to hook into the model ('before_lstm', 'after_lstm', 'before_output').
        """
        activations = []
        all_attributions = []
        # self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.model.__class__.__name__}")

        # Register hooks at different stages based on the version
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # For LSTM, handle tuple outputs
                activations.append(output.detach())

            return hook

        # Set hooks based on the chosen version
        if self.hook_location == 'input':
            # Hook into the input before LSTM (you may need to adjust depending on your model architecture)
            for name, layer in self.model.named_modules():
                if isinstance(layer, torch.nn.LSTM):  # Assuming LSTM is the first main layer
                    layer.register_forward_hook(save_activation(name))  # Use forward hook instead of forward pre-hook
                    break
        elif self.hook_location == 'lstm':
            # Hook into the output of the LSTM layer
            for name, layer in self.model.named_modules():
                if isinstance(layer, torch.nn.LSTM):
                    layer.register_forward_hook(save_activation(name))
                    break
        elif self.hook_location == 'penultimate':
            # Hook into fc1 (penultimate layer)
            for name, layer in self.model.named_modules():
                if name == "fc5":  # Specifically target fc1
                    layer.register_forward_hook(save_activation(name))
                    break
        else:
            raise ValueError(f"Unknown hook location: {self.hook_location}")

        for inputs, _ in valid_loader:
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)

                # Compute output importance for the batch
                if outputs.dim() == 3:
                    output_importance = outputs.mean(dim=1)  # Mean over time steps
                elif outputs.dim() == 2:
                    output_importance = outputs
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                reduced_output_importance = output_importance[:, :len(self.selected_features)]

                # Iterate over samples in the batch
                for i in range(inputs.size(0)):
                    sample_importance = torch.zeros(len(self.selected_features), device=self.device)

                    for activation in activations:
                        if activation.dim() == 3:  # If activations have time steps
                            layer_importance = activation[i].sum(dim=0)[:len(self.selected_features)]
                        elif activation.dim() == 2:
                            layer_importance = activation[i][:len(self.selected_features)]

                        sample_importance += layer_importance * reduced_output_importance[i]

                    all_attributions.append(sample_importance.cpu().numpy())  # Store sample-level attribution

        all_attributions = np.array(all_attributions)  # Shape: [num_samples, num_features]

        print(f"Final shape of all_attributions: {all_attributions.shape}")

        # # Apply ACTIF variant for aggregation
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)

        print(f"Calculated importance shape: {importance.shape}")

        # Ensure that the importance variable is a list or array with the same length as the number of features
        if not isinstance(importance, np.ndarray):
            importance = np.array(importance)

        if importance.shape[0] != len(self.selected_features):
            raise ValueError(
                f"ACTIF method returned {importance.shape[0]} importance scores, but {len(self.selected_features)} features are expected."
            )

        # Prepare the results with the aggregated importance
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]

        return results

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
