import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# from scipy.stats import median_absolute_deviation
# Note: If you prefer, you can also use a custom robust estimator.

# -------------------------
# Singleton Metaclass
# -------------------------
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# -------------------------
# Custom Pooling Layer (for demonstration)
# -------------------------
class MaxOverTimePooling(nn.Module):
    def __init__(self):
        super(MaxOverTimePooling, self).__init__()
    def forward(self, x):
        # x: (batch, timesteps, features)
        return x.max(dim=1)[0]  # returns (batch, features)

# -------------------------
# Bayesian Feature Importance Class
# -------------------------
# Define a helper function for the median absolute deviation (MAD)
def median_absolute_deviation(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

class A6_BayesianInference_Optim(metaclass=Singleton):
    def __init__(self, model, features, device, use_gaussian_spread=True,
                 initial_dataloader=None, initial_alpha=0.1, decay_rate=0.01):
        """
        Initialize the BayesianFeatureImportance analyzer.

        Args:
            model (torch.nn.Module): The trained model.
            features (list): List of feature names.
            device (str): Device (e.g., "cuda" or "cpu").
            use_gaussian_spread (bool): Whether to use Gaussian spreading in backprop.
            initial_dataloader (DataLoader, optional): A small sample dataloader to initialize priors.
            initial_alpha (float): Initial learning rate for Bayesian updates.
            decay_rate (float): Decay rate for the learning rate.
        """
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.use_gaussian_spread = use_gaussian_spread

        self.initial_alpha = initial_alpha
        self.decay_rate = decay_rate
        self.update_count = 0

        # Initialize priors using a small sample if available
        if initial_dataloader is not None:
            raw_importances = self.forward_pass_with_importance_from_dataloader(initial_dataloader, return_raw=True)
            # Aggregate over timesteps per sample (using median for robustness)
            aggregated = np.median(raw_importances.cpu().detach().numpy(), axis=1)  # shape: (num_samples, num_features)
            self.mu_prior = np.median(aggregated, axis=0)
            # Use MAD as robust estimate for observational variance; multiply by constant to approximate STD
            def robust_std(x):
                mad = median_absolute_deviation(x)
                return mad * 1.4826  # constant for normal distribution
            sigma_est = np.apply_along_axis(robust_std, 0, aggregated)
            sigma_est[sigma_est < 1e-3] = 1e-3  # enforce minimum variance
            self.sigma_prior = sigma_est
        else:
            self.mu_prior = np.zeros(len(self.selected_features))
            self.sigma_prior = np.ones(len(self.selected_features))

        # Containers for hooks:
        self.activations = {}
        self.layer_types = {}
        self.max_indices = {}
        self.register_hooks()

    def register_hooks(self):
        """Register forward hooks on relevant layers: Linear, LSTM, and our custom pooling."""
        def hook_fn(name, module):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
                self.layer_types[name] = type(module).__name__
                if isinstance(module, MaxOverTimePooling):
                    # For pooling layers, store the indices used in the max operation.
                    _, max_idx = torch.max(output, dim=1)
                    self.max_indices[name] = max_idx
            return hook
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.LSTM, MaxOverTimePooling)):
                layer.register_forward_hook(hook_fn(name, layer))

    def forward_pass_with_importance(self, inputs, return_raw=False):
        """
        Perform a forward pass and accumulate absolute activations ("raw importances")
        from key layers. Interpolates outputs to match the number of selected features.

        Args:
            inputs (torch.Tensor): Input tensor (batch, timesteps, ...).
            return_raw (bool): If True, returns raw importances for each timestep.

        Returns:
            torch.Tensor: Aggregated importances (batch, num_features) or raw (batch, timesteps, num_features).
        """
        self.model.eval()
        activations = inputs
        T = inputs.shape[1]
        accumulated_importance = None

        # Process through each child layer of the model
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.LSTM):
                activations, _ = layer(activations)
                layer_importance = torch.abs(activations)  # (batch, L, hidden)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)
                if activations.dim() == 2:
                    # Expand 2D output to 3D to match timesteps
                    layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
                else:
                    layer_importance = torch.abs(activations)
            else:
                continue  # Skip other layers

            # Interpolate along feature dimension to match selected features count
            N, T_current, orig_features = layer_importance.shape
            reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            interpolated = F.interpolate(reshaped, size=len(self.selected_features), mode="linear", align_corners=False)
            interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
            if accumulated_importance is None:
                accumulated_importance = interpolated_importance
            else:
                if accumulated_importance.shape != interpolated_importance.shape:
                    raise ValueError(f"Shape mismatch: {accumulated_importance.shape} vs {interpolated_importance.shape}")
                accumulated_importance += interpolated_importance

        if return_raw:
            return accumulated_importance
        else:
            # Aggregate over timesteps (mean)
            aggregated_importance = accumulated_importance.mean(dim=1)
            return aggregated_importance

    def forward_pass_with_importance_from_dataloader(self, dataloader, return_raw=False):
        """Computes raw importances over a dataloader."""
        all_importances = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)
            imp = self.forward_pass_with_importance(batch, return_raw=return_raw)
            all_importances.append(imp)
        return torch.cat(all_importances, dim=0)

    def analyze(self, dataloader, return_raw=False):
        """
        Compute importances over the entire dataloader.

        Args:
            dataloader: DataLoader.
            return_raw (bool): If True, return raw (timesteps) importances.

        Returns:
            torch.Tensor: Aggregated importances.
        """
        all_importances = []
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)
            imp = self.forward_pass_with_importance(batch, return_raw=return_raw)
            if imp is None or torch.isnan(imp).any():
                print(f"⚠️ Skipping batch {batch_idx} due to NaNs.")
                continue
            all_importances.append(imp)
            cumulative = torch.cat(all_importances, dim=0)
            print(f"After batch {batch_idx}, cumulative shape: {cumulative.shape}")
        if not all_importances:
            raise ValueError("No valid importances computed!")
        return torch.cat(all_importances, dim=0)

    def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
        """
        Computes the feature importance for each feature from aggregated importances
        (calculated over all samples) using the inverted weighted mean.

        Args:
            activation (np.ndarray): Array of shape (num_samples, num_features).

        Returns:
            np.ndarray: Adjusted importance vector (num_features,).
        """
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)
        # Normalize the mean
        normalized_mean = (mean_activation - np.min(mean_activation)) / (np.max(mean_activation) - np.min(mean_activation) + epsilon)
        # Invert the normalized std (low variability rewarded)
        inverse_normalized_std = 1 - ((std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation) + epsilon))
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance

    def most_important_timestep(self, all_attributions):
        """Returns indices (per sample, per feature) of the timestep with maximum absolute importance."""
        return np.argmax(np.abs(all_attributions), axis=1)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
        """
        Aggregates raw importances (over timesteps) using an inverted weighted mean.

        Args:
            all_attributions (np.ndarray): Shape (num_samples, timesteps, num_features).

        Returns:
            np.ndarray: Aggregated importances (num_samples, num_features).
        """
        mean_activation = np.mean(np.abs(all_attributions), axis=1)
        std_activation = np.std(np.abs(all_attributions), axis=1)
        min_mean = np.min(mean_activation, axis=1, keepdims=True)
        max_mean = np.max(mean_activation, axis=1, keepdims=True)
        normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
        min_std = np.min(std_activation, axis=1, keepdims=True)
        max_std = np.max(std_activation, axis=1, keepdims=True)
        normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
        inverse_normalized_std = 1 - normalized_std
        aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
        return aggregated_importance

    def aggregate_importances_over_samples(self, sample_importances):
        """
        Aggregates per-sample importances (shape: [num_samples, num_features])
        to produce a global importance vector (num_features,).

        Returns:
            np.ndarray: Global importance (num_features,).
        """
        global_importance = np.mean(sample_importances, axis=0)
        return global_importance

    def aggregate_and_return(self, sample_importances):
        """
        Aggregates the computed importances (from batches) and returns them in a SHAP-like format.

        Args:
            sample_importances (List[torch.Tensor]): List of per-batch importance tensors.

        Returns:
            List[dict]: List of dictionaries with 'feature', 'attribution'.
        """
        if isinstance(sample_importances, list) and len(sample_importances) == 0:
            raise ValueError("No feature importances calculated from the data.")
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
                           for imp in sample_importances]
        all_attributions = np.vstack(all_importances)
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in importances. Replacing with zeros.")
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)
        print("All Attributions shape:", all_attributions.shape)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        results = [{'feature': self.selected_features[i], 'attribution': all_attributions[i]}
                   for i in range(len(self.selected_features))]
        print("Results from Aggregation:", results)
        return results

    # ----------------------------
    # Bayesian Updating
    # ----------------------------
    def update_bayesian(self, sample_importance, sigma_obs):
        """
        Updates the Bayesian priors (mu_prior and sigma_prior) for each feature based on a new observation.

        Args:
            sample_importance (np.ndarray): Aggregated importance vector (num_features,).
            sigma_obs (np.ndarray): Observational uncertainty (per feature, shape: num_features,).
        """
        prior_var = self.sigma_prior ** 2
        obs_var = sigma_obs ** 2
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (self.mu_prior / prior_var + sample_importance / obs_var)
        current_alpha = self.initial_alpha / (1 + self.update_count * self.decay_rate)
        self.mu_prior = (1 - current_alpha) * self.mu_prior + current_alpha * posterior_mean
        # For sigma, we combine the variances via an exponential moving average:
        self.sigma_prior = np.sqrt((1 - current_alpha) * (self.sigma_prior ** 2) + current_alpha * posterior_var)
        self.update_count += 1
        print("Updated Bayesian Priors:")
        print("Mu:", self.mu_prior)
        print("Sigma:", self.sigma_prior)

    def analyze_bayesian(self, dataloader, device):
        """
        Processes the entire dataloader and updates Bayesian priors for each sample.
        For each sample, the observational uncertainty (σ_obs) is computed as the robust std (MAD)
        over timesteps. Then the Bayesian update is applied.

        Returns:
            Tuple: (mu_prior, sigma_prior)
        """
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, dtype=torch.float32)
            raw = self.forward_pass_with_importance(batch, return_raw=True)
            # Aggregate over timesteps using median for robustness:
            sample_imp = np.median(raw.cpu().detach().numpy(), axis=1)  # (batch, num_features)
            # Compute observational uncertainty using robust MAD over timesteps:
            sigma_obs = np.apply_along_axis(lambda x: median_absolute_deviation(x) * 1.4826,
                                            1, np.abs(raw.cpu().detach().numpy()))
            sigma_obs[sigma_obs < 0.01] = 0.01
            for s in range(sample_imp.shape[0]):
                self.update_bayesian(sample_imp[s], sigma_obs[s])
            print(f"After batch {batch_idx}: mu_prior: {self.mu_prior}, sigma_prior: {self.sigma_prior}")
        print("Final Bayesian mu_prior:", self.mu_prior)
        print("Final Bayesian sigma_prior:", self.sigma_prior)
        return self.mu_prior, self.sigma_prior

    def compute_bayesACTIF(self, valid_loader, combine_uncertainty=False, combine_method='None'):
        """
        Performs the Bayesian DeepACTIF analysis: iterates over the validation data,
        updates the Bayesian priors, and then returns final attributions and uncertainties.
        Optionally, the final score can combine attribution and uncertainty.

        Args:
            valid_loader: DataLoader for validation data.
            combine_uncertainty (bool): If True, combine attribution and uncertainty.
            combine_method (str): 'division' or 'multiplication' to combine them.

        Returns:
            List[dict]: List with keys 'feature', 'attribution', and 'uncertainty'.
        """
        mu_post, sigma_post = self.analyze_bayesian(valid_loader, self.device)
        results = []
        for i, feature in enumerate(self.selected_features):
            attribution = mu_post[i]
            uncertainty = sigma_post[i]
            if combine_uncertainty:
                if combine_method == 'division':
                    final_score = attribution / uncertainty if uncertainty != 0 else attribution
                elif combine_method == 'multiplication':
                    final_score = attribution * uncertainty
                else:
                    final_score = attribution
            else:
                final_score = attribution
            results.append({
                'feature': feature,
                'attribution': final_score,
                'uncertainty': uncertainty
            })
        print("Bayesian ACTIF Results:")
        for r in results:
            print(r)
        return results

    # ----------------------------
    # Backpropagation Functions
    # ----------------------------
    def backpropagate_linear(self, importance, activations, layer_name):
        """
        Backpropagates importance through a Linear layer via interpolation.
        """
        print(f"🔄 Backpropagating through Linear layer: {layer_name}")
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
        if importance.dim() == 2:
            importance = importance.unsqueeze(1)
        target_size = (activations.shape[-1],)
        print(f"⚠ Interpolating Linear layer from {importance.shape[-1]} to {target_size}")
        importance = F.interpolate(importance, size=target_size, mode="linear", align_corners=False).squeeze(1)
        print(f"✅ Backpropagated Linear Importance Shape: {importance.shape}")
        return importance

    def backpropagate_lstm(self, importance, activations, layer_name):
        """
        Backpropagates importance through an LSTM layer using bilinear interpolation.
        """
        print(f"🔄 Backpropagating through LSTM layer: {layer_name}")
        batch_size, timesteps, num_features = activations.shape
        if importance.dim() == 2:
            importance = importance.unsqueeze(1)