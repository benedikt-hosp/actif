import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


#
# Zuerst definieren wir einen Singleton-Metaklassen-Mechanismus:
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Erzeuge eine neue Instanz, wenn sie noch nicht existiert
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Beispiel: BayesianFeatureImportance als Singleton
class A5_BayesianInference(metaclass=Singleton):
    def __init__(self, model, features, device, use_gaussian_spread=True):
        """
        Initialize the analyzer.

        Args:
            model (torch.nn.Module): The trained model.
            features (list): List of feature names.
            device (str): The device to run the analysis on.
            use_gaussian_spread (bool): Flag to toggle between Gaussian spreading and equal distribution.
        """
        self.use_gaussian_spread = True
        self.max_indices = None
        self.layer_types = None
        self.activations = None
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.alpha = 0.1
        # Beispielhafte Initialisierung der Priorwerte (diese werden dann über alle Subjects hinweg aktualisiert)
        self.mu_prior = np.zeros(len(self.selected_features))
        self.sigma_prior = np.ones(len(self.selected_features))

        # Bayesian-Prior für jedes Feature (als Vektor, Länge = Anzahl Features)
        self.mu_prior = np.zeros(len(self.selected_features))
        self.sigma_prior = np.ones(len(self.selected_features))

    def register_hooks(self):
        def hook_fn(name, module):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
                self.layer_types[name] = type(module).__name__
                if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                    _, max_idx = torch.max(output, dim=1)
                    self.max_indices[name] = max_idx

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                layer.register_forward_hook(hook_fn(name, layer))

    def backpropagate_max_pooling(self, importance, activations, layer_name):
        print(f"🔄 Backpropagating through Max Pooling layer: {layer_name}")
        if layer_name not in self.max_indices:
            raise ValueError(f"🚨 No max_indices found for {layer_name}.")
        max_indices = self.max_indices[layer_name]
        batch_size, timesteps, num_features = activations.shape
        expanded_importance = torch.zeros_like(activations)
        if self.use_gaussian_spread:
            for b in range(batch_size):
                for f in range(num_features):
                    max_t = max_indices[b, f].item()
                    for t in range(timesteps):
                        expanded_importance[b, t, f] = (
                                importance[b, f] * torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
                                                                                   device=importance.device) - max_t) / 2.0) ** 2)
                        )
        else:
            for b in range(batch_size):
                for f in range(num_features):
                    expanded_importance[b, max_indices[b, f], f] = importance[b, f]
        print(f"✅ Final importance shape after Max Pooling: {expanded_importance.shape}")
        return expanded_importance

    def forward_pass_with_importance(self, inputs, return_raw=False):
        self.model.eval()
        activations = inputs
        T = inputs.shape[1]
        accumulated_importance = None

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.LSTM):
                activations, _ = layer(activations)
                layer_importance = torch.abs(activations)  # (batch, L, hidden)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)
                if activations.dim() == 2:
                    layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
                else:
                    layer_importance = torch.abs(activations)
            else:
                continue

            # Interpolation entlang der Feature-Dimension
            # N, T_current, orig_features = layer_importance.shape
            # reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            # interpolated = F.interpolate(reshaped, size=len(self.selected_features),
            #                              mode="linear", align_corners=False)
            # interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))

            N, T_current, orig_features = layer_importance.shape
            reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            interpolated = F.interpolate(
                reshaped, size=(len(self.selected_features),), mode="linear", align_corners=False
            )
            interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))

            assert interpolated_importance.shape[-1] == len(
                self.selected_features), "Interpolation hat zu viele Features!"

            if accumulated_importance is None:
                accumulated_importance = interpolated_importance
            else:
                if accumulated_importance.shape != interpolated_importance.shape:
                    raise ValueError(
                        f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}")
                accumulated_importance += interpolated_importance


        if return_raw:
            return accumulated_importance
        else:
            aggregated_importance = accumulated_importance.mean(dim=1)
            return aggregated_importance

    def analyze(self, dataloader, return_raw=False):
        all_importances = []
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)
            importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
            if importance is None or torch.isnan(importance).any():
                print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
                continue
            all_importances.append(importance)

            # Live-Update: Ausgabe der kumulativen Importances
            cumulative = torch.cat(all_importances, dim=0)
            print(f"Nach Batch {batch_idx}: Kumulierte Importances-Shape: {cumulative.shape}")

        if not all_importances:
            raise ValueError("❌ No valid feature importances calculated!")
        concatenated = torch.cat(all_importances, dim=0)
        return concatenated if return_raw else concatenated

    def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation) + epsilon)
        inverse_normalized_std = 1 - ((std_activation - np.min(std_activation)) / (
                np.max(std_activation) - np.min(std_activation) + epsilon))
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance

    def most_important_timestep(self, all_attributions):
        return np.argmax(np.abs(all_attributions), axis=1)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
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
        if isinstance(sample_importances, torch.Tensor):
            sample_importances = sample_importances.cpu().detach().numpy()
        print("Input to aggregate_importances_over_samples, shape:", sample_importances.shape)
        global_importance = np.mean(sample_importances, axis=0)
        print("Global importance after aggregation:", global_importance.shape, global_importance)
        if global_importance.shape[0] != len(self.selected_features):
            raise ValueError(f"Expected {len(self.selected_features)} features, but got {global_importance.shape[0]}.")
        df_final = pd.DataFrame({
            "feature": self.selected_features,
            "attribution": global_importance
        })
        print("Final DataFrame shape:", df_final.shape)
        return df_final

    def aggregate_and_return(self, sample_importances):
        if (isinstance(sample_importances, list) and len(sample_importances) == 0) or \
                (isinstance(sample_importances, torch.Tensor) and sample_importances.numel() == 0):
            raise ValueError("❌ No feature importances calculated from the data.")
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
                           for imp in sample_importances]
        all_attributions = np.vstack(all_importances)
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in all_attributions!")
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)
        print("All Attributions: ", all_attributions)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]
        print("Results from Class: ", results)
        return results

    def backpropagate_linear(self, importance, activations, layer_name):
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
        print(f"🔄 Backpropagating through LSTM layer: {layer_name}")
        batch_size, timesteps, num_features = activations.shape
        if importance.dim() == 2:
            importance = importance.unsqueeze(1)
        elif importance.dim() == 3 and importance.shape[1] == num_features:
            importance = importance.permute(0, 2, 1)
        print(f"✅ Adjusted importance shape before interpolation: {importance.shape}")
        importance = importance.unsqueeze(1)
        target_size = (timesteps, len(self.selected_features))
        print(f"✅ Interpolating to match input shape {target_size}")
        importance = F.interpolate(importance, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
        print(f"✅ Final importance shape after LSTM backpropagation: {importance.shape}")
        return importance

    def weighted_interpolation(self, importance, activations):
        batch_size, timesteps, num_features = activations.shape
        max_indices = activations.abs().max(dim=1).indices
        spread_importance = torch.zeros_like(activations)
        for i in range(batch_size):
            max_t = max_indices[i].item()
            time_range = torch.arange(timesteps, device=importance.device)
            gaussian_weights = torch.exp(-((time_range - max_t) ** 2) / (2 * (timesteps / 5) ** 2))
            gaussian_weights = gaussian_weights / gaussian_weights.sum()
            spread_importance[i] = gaussian_weights[:, None] * importance[i]
        print(f"✅ Final importance shape after max-pooling backpropagation: {spread_importance.shape}")
        return spread_importance

        # --- Bayesian Updating ---
    # def update_bayesian(self, sample_importance, sigma_obs=None):
    #     """
    #     Updates the Bayesian prior per feature, using feature-wise uncertainty updating.
    #
    #     Args:
    #         sample_importance (np.ndarray): Feature importance for one sample, shape (num_features,).
    #         sigma_obs (np.ndarray or float, optional): Observational uncertainty per feature.
    #             - If float, applies same sigma_obs to all features.
    #             - If array, must have shape (num_features,).
    #     """
    #     print("\n🔍 Before update:")
    #     print(f"Mu_prior: {self.mu_prior}")
    #     print(f"Sigma_prior: {self.sigma_prior}")
    #
    #     if sigma_obs is None:
    #         sigma_obs = np.ones_like(self.sigma_prior) * 0.1  # Default: Same uncertainty per feature
    #     elif isinstance(sigma_obs, float):
    #         sigma_obs = np.ones_like(self.sigma_prior) * sigma_obs  # Convert to per-feature array
    #
    #     # Convert variances (σ²)
    #     prior_var = self.sigma_prior ** 2  # Current prior variance (per feature)
    #     obs_var = sigma_obs ** 2  # Observation variance (per feature)
    #
    #     # Bayesian Update Equations
    #     posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    #     posterior_mean = posterior_var * (self.mu_prior / prior_var + sample_importance / obs_var)
    #
    #     # Update priors
    #     self.mu_prior = posterior_mean
    #     self.sigma_prior = np.sqrt(posterior_var)  # Convert back to standard deviation
    #
    #     print("\n✅ After update:")
    #     print(f"Mu_prior: {self.mu_prior}")
    #     print(f"Sigma_prior: {self.sigma_prior}")  # Now each feature should have a unique value!

    def update_bayesian(self, sample_importance, sigma_obs=None):
        """
        Vectorized Bayesian Update for Feature Importances.

        Args:
            sample_importance (np.ndarray): Importance values for one sample (num_features,).
            sigma_obs (np.ndarray or float, optional): Observation uncertainty per feature.
                - If float, applies same sigma_obs to all features.
                - If array, must have shape (num_features,).
        """
        if sigma_obs is None:
            sigma_obs = np.ones_like(self.sigma_prior) * 0.1  # Default uncertainty
        elif isinstance(sigma_obs, float):
            sigma_obs = np.ones_like(self.sigma_prior) * sigma_obs  # Broadcast to array

        # Convert standard deviations to variances
        prior_var = self.sigma_prior ** 2
        obs_var = sigma_obs ** 2

        # Compute posterior using vectorized operations
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (self.mu_prior / prior_var + sample_importance / obs_var)

        # Update priors
        self.mu_prior = posterior_mean
        self.sigma_prior = np.sqrt(posterior_var)  # Convert back to std deviation

    # def analyze_bayesian(self, dataloader):
    #     """
    #     Computes Bayesian feature importances and updates priors per feature.
    #
    #     Returns:
    #         - np.ndarray: Posterior means (`mu_prior`)
    #         - np.ndarray: Posterior standard deviations (`sigma_prior`)
    #     """
    #     print("🔍 Starting Bayesian Analysis...")
    #
    #     for batch_idx, batch in enumerate(dataloader):
    #         if isinstance(batch, (list, tuple)):
    #             batch = batch[0]
    #         batch = batch.to(self.device, dtype=torch.float32)
    #
    #         # Compute feature importance for this batch
    #         sample_importances = self.forward_pass_with_importance(batch, return_raw=False)
    #         sample_importances_np = sample_importances.cpu().detach().numpy()  # Shape: (batch_size, num_features)
    #
    #         for i, sample_imp in enumerate(sample_importances_np):
    #             print(f"🛠 Updating sample {i} in batch {batch_idx}")
    #
    #             # 🔥 Compute per-feature observational uncertainty dynamically!
    #             sigma_obs = np.std(sample_importances_np, axis=0) + 0.01  # Add small value to avoid zero variance
    #
    #             self.update_bayesian(sample_imp, sigma_obs=sigma_obs)  # 🔄 Feature-wise Bayesian update
    #
    #     print("✅ Final Bayesian priors computed.")
    #     return self.mu_prior, self.sigma_prior

    def analyze_bayesian(self, dataloader):
        """
        Computes Bayesian feature importances and updates priors per feature.

        Returns:
            - np.ndarray: Posterior means (`mu_prior`)
            - np.ndarray: Posterior standard deviations (`sigma_prior`)
        """
        print("🔍 Starting Bayesian Analysis...")

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)

            # Compute feature importance for this batch
            sample_importances = self.forward_pass_with_importance(batch, return_raw=False)
            sample_importances_np = sample_importances.cpu().detach().numpy()  # Shape: (batch_size, num_features)

            # 🔥 Compute per-feature observational uncertainty dynamically!
            sigma_obs = np.std(sample_importances_np, axis=0) + 0.01  # Add small value to avoid zero variance

            # 🔄 Apply **vectorized** Bayesian update for all samples at once
            for sample_imp in sample_importances_np:
                self.update_bayesian(sample_imp, sigma_obs=sigma_obs)

        print("✅ Final Bayesian priors computed.")
        return self.mu_prior, self.sigma_prior

    def compute_bayesACTIF(self, valid_loader):
        """
        Führt Bayesian Feature-Attributions-Analyse durch und gibt eine Liste mit
        Feature-Namen, Attributionswerten und Unsicherheiten zurück.

        Returns:
            List[dict]: [{feature, attribution, uncertainty}]
        """
        mu_post, sigma_post = self.analyze_bayesian(valid_loader)

        # Ausgabe als Liste von Dictionaries mit genau EINEM Eintrag pro Feature!
        results = [
            {"feature": self.selected_features[i], "attribution": mu_post[i], "uncertainty": sigma_post[i]}
            for i in range(len(self.selected_features))
        ]

        print(f"🔍 Mu_prior shape: {mu_post.shape}")  # Sollte (num_features,) sein
        print(f"🔍 Sigma_prior shape: {sigma_post.shape}")  # Sollte (num_features,) sein

        print("Features length: ", len(self.selected_features))
        assert mu_post.shape[0] == len(self.selected_features), "Aggregation falsch: Mu_post hat zu viele Werte!"
        assert sigma_post.shape[0] == len(self.selected_features), "Aggregation falsch: Sigma_post hat zu viele Werte!"

        results = [
            {"feature": self.selected_features[i],
             "attribution": mu_post[i],
             "uncertainty": sigma_post[i]}
            for i in range(len(self.selected_features))
        ]

        return results

