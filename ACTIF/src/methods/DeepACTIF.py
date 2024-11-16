import torch
import torch.nn.functional as F
import numpy as np


# Set seeds for reproducibility
# torch.manual_seed(0)
# np.random.seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)
#     torch.cuda.manual_seed_all(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

class DeepACTIF:
    def __init__(self, model, selected_features, target_layer, device='cuda'):
        """
        Initialize DeepACTIF for a single target layer.

        Args:
            model: The PyTorch model to analyze.
            selected_features: List of features to consider for attribution.
            target_layer: Name of the single target layer for capturing activations.
            device: Device to use ('cuda' or 'cpu').
        """
        self.model = model
        self.selected_features = selected_features
        self.target_layer = target_layer  # Single layer
        self.device = device
        self.activations = []  # Store activations for the single layer

    def _hook_layer(self):
        """Register a hook to capture activations at the target layer."""
        for name, layer in self.model.named_modules():
            if name == self.target_layer:
                print(f"Registering hook for layer: {name}")  # Debug message
                layer.register_forward_hook(self._save_activation)
                return
        raise ValueError(f"Layer '{self.target_layer}' not found in the model.")

    def _save_activation(self, module, input, output):
        """Hook function to capture the activations."""
        if isinstance(output, tuple):  # Handle tuple output, e.g., from LSTM
            output = output[0]
        self.activations.append(output.detach())  # Detach to avoid gradient tracking

    def compute_deepactif(self, dataloader):
        """
        Compute DeepACTIF importance scores for a single layer.

        Args:
            dataloader: DataLoader containing validation or test data.

        Returns:
            all_attributions: Numpy array of feature attributions.
        """
        self.model.to(self.device)
        self.model.eval()

        self.activations = []  # Clear activations for fresh computation
        self._hook_layer()  # Set up the hook

        # Run the model to capture activations
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                self.model(inputs)  # Forward pass triggers hooks

        # Stack activations from all batches
        activations_stacked = torch.cat(self.activations, dim=0)

        # Reduce over time steps if necessary
        if activations_stacked.dim() == 3:  # [samples, timesteps, features]
            activations_stacked = activations_stacked.mean(dim=1)

        # Match the number of features if necessary
        if activations_stacked.shape[1] != len(self.selected_features):
            all_attributions = F.interpolate(
                activations_stacked.unsqueeze(1),
                size=len(self.selected_features),
                mode="linear",
                align_corners=True
            ).squeeze(1).cpu().numpy()
        else:
            all_attributions = activations_stacked.cpu().numpy()

        print(f"Attributions computed with shape: {all_attributions.shape}")
        return all_attributions

    # def compute_deepactif(self, dataloader, inactivate_layer=None, actif_variant='mean'):
    #     """
    #     Computes feature importance by capturing activations at specified layers and averaging across timesteps.
    #
    #     Args:
    #         dataloader: The DataLoader containing validation or test data.
    #         inactivate_layer: The layer name for which activations should be processed.
    #         actif_variant: Variant of ACTIF to use in importance calculation.
    #     """
    #     self.model.to(self.device)
    #     self.model.eval()  # Ensure the model is in evaluation mode
    #
    #     # Reset activations and set up hooks
    #     self.activations = {key: [] for key in self.activations}
    #     self.setup_hooks()
    #
    #     # Capture activations
    #     with torch.no_grad():
    #         for inputs, _ in dataloader:
    #             inputs = inputs.to(self.device)
    #             self.model(inputs)  # Forward pass to trigger hooks
    #
    #     # Stack activations from all batches
    #     activations_stacked = {
    #         key: torch.cat(self.activations[key], dim=0)
    #         for key in self.activations if self.activations[key]
    #     }
    #
    #     if inactivate_layer is None:
    #         inactivate_layer = self.target_layers[0]  # Default to the first layer if unspecified
    #
    #     if inactivate_layer not in activations_stacked:
    #         raise ValueError(
    #             f"Layer '{inactivate_layer}' not found in activations. Available layers: {list(activations_stacked.keys())}"
    #         )
    #
    #     # Extract activations for the specified layer
    #     layer_activations = activations_stacked[inactivate_layer]
    #
    #     # Step 1: Average across timesteps
    #     if layer_activations.dim() == 3:  # [samples, timesteps, features]
    #         layer_activations = layer_activations.mean(dim=1)  # Average over timesteps
    #
    #     # Step 2: Resize to match the number of features if necessary
    #     if layer_activations.shape[1] != len(self.selected_features):
    #         all_attributions = F.interpolate(
    #             layer_activations.unsqueeze(1),
    #             size=len(self.selected_features),
    #             mode="linear",
    #             align_corners=True
    #         ).squeeze(1).cpu().numpy()
    #     else:
    #         all_attributions = layer_activations.cpu().numpy()
    #
    #     return all_attributions
    # Step 3: Pass

    # mean over batches not samples
    # def compute_deepactif(self, dataloader, inactivate_layer=None, actif_variant='inv'):
    #     """
    #     Computes feature importance by capturing activations at specified layers.
    #
    #     Args:
    #         dataloader: The DataLoader containing validation or test data.
    #         inactivate_layer: The layer name for which activations should be processed.
    #         actif_variant: Variant of ACTIF to use in importance calculation.
    #     """
    #     self.model.to(self.device)
    #     self.model.eval()  # Ensure the model is in evaluation mode
    #
    #     # Reset activations and set up hooks
    #     self.activations = {key: [] for key in self.activations}
    #     self.setup_hooks()
    #
    #     # Capture activations
    #     with torch.no_grad():
    #         for inputs, _ in dataloader:
    #             inputs = inputs.to(self.device)
    #             self.model(inputs)  # Forward pass to trigger hooks
    #
    #     # Stack activations from each batch and process the specified layer
    #     activations_stacked = {
    #         key: torch.cat(self.activations[key], dim=0)
    #         for key in self.activations if self.activations[key]
    #     }
    #
    #     if inactivate_layer is None:
    #         inactivate_layer = self.target_layers[0]  # Default to the first layer if unspecified
    #
    #     # should noe be [samples,timesteps, features]
    #     # Aggregate activations over time if needed
    #     if inactivate_layer not in activations_stacked:
    #         raise ValueError(
    #             f"Layer '{inactivate_layer}' not found in activations. Available layers: {list(activations_stacked.keys())}")
    #
    #     if activations_stacked[inactivate_layer].dim() == 3:
    #         activations_stacked[inactivate_layer] = activations_stacked[inactivate_layer].sum(dim=1)
    #
    #     # should noe be [samples, features]
    #     # Resize activations to match input feature dimensions if necessary
    #     if activations_stacked[inactivate_layer].shape[1] != len(self.selected_features):
    #         all_attributions = F.interpolate(
    #             activations_stacked[inactivate_layer].unsqueeze(1),
    #             size=len(self.selected_features),
    #             mode="linear",
    #             align_corners=True
    #         ).squeeze(1).cpu().numpy()
    #     else:
    #         all_attributions = activations_stacked[inactivate_layer].cpu().numpy()
    #
    #     print("Attribution of DeepACTIF has shape: ", all_attributions.shape)
    #     return all_attributions
