import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class A1_AutoDeepACTIF_AnalyzerBackprop:
    def __init__(self, model, features, device, use_gaussian_spread=True):
        """
        Initialize the analyzer.
        
        Args:
            model (torch.nn.Module): The trained model.
            features (list): List of feature names.
            device (str): The device to run the analysis on.
            use_gaussian_spread (bool): Flag to toggle between Gaussian spreading and equal distribution.
        """
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.use_gaussian_spread = use_gaussian_spread  # 🔹 Flag to choose backprop method

        self.activations = {}
        self.layer_types = {}
        self.max_indices = {}  # 🔹 Store max-pooling indices
        self.register_hooks()

    def register_hooks(self):
        """Registers forward hooks to store activations and max-pooling indices."""

        def hook_fn(name, module):
            def hook(module, input, output):
                if isinstance(output, tuple):  # Handle LSTM outputs
                    output = output[0]
                self.activations[name] = output.detach()
                self.layer_types[name] = type(module).__name__

                # 🔹 Store max-pooling indices correctly
                if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                    _, max_idx = torch.max(output, dim=1)  # Shape (batch, features)
                    self.max_indices[name] = max_idx  # 🔹 Save indices by layer name

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                layer.register_forward_hook(hook_fn(name, layer))

    def backpropagate_max_pooling(self, importance, activations, layer_name):
        """
        Backpropagates importance through the Max Pooling layer.
        - `importance`: Shape (batch, features)
        - `activations`: Shape (batch, timesteps, features)
        """
        print(f"🔄 Backpropagating through Max Pooling layer: {layer_name}")

        # Ensure `max_indices` is available
        if layer_name not in self.max_indices:
            raise ValueError(
                f"🚨 No max_indices found for {layer_name}. Make sure max-pooling layers are hooked correctly!")

        max_indices = self.max_indices[layer_name]  # Get stored indices (batch, features)
        batch_size, timesteps, num_features = activations.shape

        # Initialize backpropagated importance (zeros everywhere)
        expanded_importance = torch.zeros_like(activations)

        if self.use_gaussian_spread:
            # 🚀 Apply Gaussian weighting centered at max indices
            for b in range(batch_size):
                for f in range(num_features):
                    max_t = max_indices[b, f].item()  # Extract max timestep
                    for t in range(timesteps):
                        expanded_importance[b, t, f] = (
                                importance[b, f] * torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
                                                                                   device=importance.device) - max_t) / 2.0) ** 2)
                        )
        else:
            # 🚀 Directly assign importance to the max-pooled timesteps
            for b in range(batch_size):
                for f in range(num_features):
                    expanded_importance[b, max_indices[b, f], f] = importance[b, f]

        print(f"✅ Final importance shape after Max Pooling: {expanded_importance.shape}")
        return expanded_importance

    def forward_pass_with_importance(self, sample):
        """
        Passes a single sample through the model and backpropagates importance.
        """
        with torch.no_grad():
            _ = self.model(sample)

        # 🔹 Start importance calculation from last layer
        last_layer_name = list(self.activations.keys())[-1]
        accumulated_importance = self.activations[last_layer_name].sum(dim=1)  # Sum over features

        # Ensure correct shape (batch, features)
        if accumulated_importance.dim() == 1:
            accumulated_importance = accumulated_importance.unsqueeze(0)  # Add batch dim

        print(f"✅ Initial importance shape from {last_layer_name}: {accumulated_importance.shape}")

        # 🔄 Backpropagate through layers
        for layer_name in reversed(list(self.activations.keys())):
            prev_activation = self.activations[layer_name]
            layer_type = self.layer_types[layer_name]

            # 🚨 Check if this is a max-pooling layer and handle it properly
            if "MaxPool" in self.layer_types[layer_name]:
                accumulated_importance = self.backpropagate_max_pooling(accumulated_importance, prev_activation,
                                                                        layer_name)
            elif "Linear" in self.layer_types[layer_name]:
                accumulated_importance = self.backpropagate_linear(accumulated_importance, prev_activation, layer_name)
            elif "LSTM" in self.layer_types[layer_name]:
                accumulated_importance = self.backpropagate_lstm(accumulated_importance, prev_activation, layer_name)

        return accumulated_importance

    def analyze(self, dataloader):
        """
        Runs DeepACTIF-NG over validation data to compute feature importance per sample.
        """
        all_importances = []

        for batch_idx, batch in enumerate(dataloader):
            batch = batch[0].to(self.device)

            for sample_idx in range(batch.shape[0]):
                sample = batch[sample_idx].unsqueeze(0)  # Ensure shape (1, timesteps, features)
                sample_importance = self.forward_pass_with_importance(sample)

                if sample_importance is not None:
                    all_importances.append(sample_importance)
                else:
                    print(f"⚠️ Skipping sample {sample_idx} in batch {batch_idx}, feature importance is None!")

        if not all_importances:
            raise ValueError("❌ No feature importances calculated. Check sample processing!")

        print("Shape of importances: ", all_importances)

        all_importances = torch.cat(all_importances, dim=0)  # Shape: (num_samples, num_features)

        print(f"✅ Final shape of all_attributions: {all_importances.shape}")
        return all_importances  # Shape: (num_samples, num_features)


        # """
        # Runs DeepACTIF-NG over validation data to compute feature importance **per sample**.
        # """
        # all_importances = []
        #
        # for batch_idx, batch in enumerate(dataloader):
        #     batch = batch[0].to(self.device)
        #     print("Batch: ", batch)
        #
        #     # 🔹 Process **each sample separately** from the batch
        #     for sample_idx in range(batch.shape[0]):
        #         sample = batch[sample_idx].unsqueeze(0)  # Ensure it's (1, timesteps, features)
        #         sample_importance = self.forward_pass_with_importance(sample)
        #
        #         if sample_importance is not None:
        #             all_importances.append(sample_importance)
        #         else:
        #             print(f"⚠️ Skipping sample {sample_idx} in batch {batch_idx}, feature importance is None!")
        #
        # if not all_importances:
        #     raise ValueError("❌ No feature importances calculated. Check sample processing!")
        #
        # # 🔹 Stack importance values **per sample** instead of per batch
        # all_importances = torch.cat(all_importances, dim=0)  # Shape: (num_samples, num_features)
        #
        # print(f"✅ Final shape of all_attributions: {all_importances.shape}")
        # return all_importances  # Shape: (num_samples, num_features)

    import numpy as np

    def most_important_timestep(self, all_attributions):
        """
        Bestimmt für jedes Sample und jedes Feature den Timestep, an dem der absolute Attributionswert maximal ist.

        Args:
            all_attributions (np.ndarray): Array mit Form (num_samples, timesteps, num_features).

        Returns:
            np.ndarray: Array der Form (num_samples, num_features) mit den Indizes des wichtigsten Timesteps.
        """
        # np.argmax wählt den Index des Maximums über die Zeitachse (axis=1)
        important_timesteps = np.argmax(np.abs(all_attributions), axis=1)
        return important_timesteps

    # Beispielaufruf:
    # Angenommen, all_attributions ist ein NumPy-Array mit Form (493, 10, 34)
    # important_ts = most_important_timestep(all_attributions)
    # print("Wichtigster Timestep pro Sample und Feature:")
    # print(important_ts)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
        """
        Aggregiert Attributionswerte über die Zeit für jedes Sample mittels invers gewichteter Methode.

        Args:
            all_attributions (np.ndarray): Array der Form (num_samples, timesteps, num_features).
            epsilon (float): Kleine Konstante zur Stabilisierung der Division.

        Returns:
            np.ndarray: Aggregierte Importance-Werte mit der Form (num_samples, num_features).
        """
        # Berechne pro Sample und Feature den Mittelwert und die Standardabweichung über die Zeit (axis=1)
        mean_activation = np.mean(np.abs(all_attributions), axis=1)  # Form: (num_samples, num_features)
        std_activation = np.std(np.abs(all_attributions), axis=1)  # Form: (num_samples, num_features)

        # Normalisierung per Sample: Wir berechnen pro Sample den Minimal- und Maximalwert
        min_mean = np.min(mean_activation, axis=1, keepdims=True)
        max_mean = np.max(mean_activation, axis=1, keepdims=True)
        normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)

        min_std = np.min(std_activation, axis=1, keepdims=True)
        max_std = np.max(std_activation, axis=1, keepdims=True)
        normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
        inverse_normalized_std = 1 - normalized_std

        # Kombiniere die beiden Maße (hier als einfacher Durchschnitt)
        aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
        return aggregated_importance

    # # Beispielaufruf:
    # aggregated_importances = aggregate_importance_inv_weighted(all_attributions)
    # print("Aggregierte Importance pro Sample (Form:", aggregated_importances.shape, ")")

    def aggregate_and_return(self, sample_importances):
        """
        Aggregiert die berechneten Feature-Importances und formatiert sie im SHAP-Stil.

        Args:
            sample_importances (List[torch.Tensor]): Liste von Feature-Importance-Vektoren pro Batch.

        Returns:
            List[List[Dict]]: Liste pro Sample, die ein Dictionary mit 'feature' und 'attribution' enthält.
        """
        # Prüfe, ob sample_importances nicht leer ist
        if (isinstance(sample_importances, list) and len(sample_importances) == 0) or \
                (isinstance(sample_importances, torch.Tensor) and sample_importances.numel() == 0):
            raise ValueError("❌ No feature importances calculated from the data.")

        # Konvertiere alle Tensoren in NumPy-Arrays (sofern nötig)
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
                           for imp in sample_importances]

        # Staple alle Ergebnisse zu einem Array (Annahme: shape (num_batches, num_features))
        all_attributions = np.vstack(all_importances)

        # Optional: Debug-Ausgabe, um zu prüfen, dass all_attributions den erwarteten Inhalt hat
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in all_attributions!")
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)

        # Aggregiere mithilfe deiner ACTIF-Methode (hier: calculate_actif_inverted_weighted_mean)
        # Hinweis: Falls du pro Sample eine Aggregation möchtest, muss die Dimension stimmen.
        # Ein einfaches Beispiel: Wir nehmen an, all_attributions hat bereits die Form (num_samples, num_features)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)

        # Erstelle das Ergebnis im SHAP-ähnlichen Format: Eine Liste von Dicts pro Sample.
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]


        # results = [
        #     [{'feature': self.selected_features[i], 'attribution': importance[sample_idx, i]}
        #      for i in range(len(self.selected_features))]
        #     for sample_idx in range(importance.shape[0])
        # ]

        return results

    def calculate_actif_inverted_weighted_mean(self, activation):
        """
        Calculate feature importance using mean activations and inverse std deviation.

        Args:
            activation (np.ndarray): Shape (num_samples, num_features).

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

    def backpropagate_linear(self, importance, activations, layer_name):
        """
        Backpropagates importance through a Linear layer.
        """
        print(f"🔄 Backpropagating through Linear layer: {layer_name}")

        # Ensure correct shape
        if importance.dim() == 1:  # If (features,), add batch dim
            importance = importance.unsqueeze(0)

        if importance.dim() == 2:  # If (batch, features), ensure shape is correct
            importance = importance.unsqueeze(1)  # Convert to (batch, 1, features)

        # Interpolate back to match previous layer shape
        target_size = (activations.shape[-1],)
        print(f"⚠ Interpolating Linear layer from {importance.shape[-1]} to {target_size}")

        importance = F.interpolate(
            importance,
            size=target_size,
            mode="linear",
            align_corners=False
        ).squeeze(1)  # Remove extra dim

        print(f"✅ Backpropagated Linear Importance Shape: {importance.shape}")
        return importance

    def backpropagate_lstm(self, importance, activations, layer_name):
        """
        Backpropagates importance through an LSTM layer.
        """
        print(f"🔄 Backpropagating through LSTM layer: {layer_name}")

        batch_size, timesteps, num_features = activations.shape  # (1, 10, 150)

        # 🚀 Ensure importance shape matches expected format (batch, 1, features)
        if importance.dim() == 2:  # If (batch, features)
            importance = importance.unsqueeze(1)  # Convert to (batch, 1, features)

        elif importance.dim() == 3 and importance.shape[1] == num_features:
            # 🔹 Detected (batch, features, 1) instead of (batch, 1, features)
            importance = importance.permute(0, 2, 1)  # Convert (batch, features, 1) → (batch, 1, features)

        print(f"✅ Adjusted importance shape before interpolation: {importance.shape}")  # Expected: (1, 1, 150)

        # 🚀 Reshape importance for interpolation
        importance = importance.unsqueeze(1)  # Add a channel dimension → (batch, C=1, 1, features)

        # 🔹 Target shape should be (batch, C=1, timesteps, features)
        #target_size = (timesteps, num_features)  # (10, 150)
        target_size = (timesteps, len(self.selected_features))  # (10, 150)
        print(f"✅ Interpolating to match input shape {target_size}")

        # 🚀 Perform interpolation correctly
        importance = F.interpolate(
            importance,  # Shape: (batch, 1, 1, features)
            size=target_size,  # Match (time_steps, features)
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # Remove channel dim → (batch, timesteps, features)

        print(f"✅ Final importance shape after LSTM backpropagation: {importance.shape}")
        return importance  # Expected: (batch, timesteps, features)

    def weighted_interpolation(self, importance, activations):
        """
        Spreads importance over timesteps using a Gaussian distribution centered at the max timestep.

        Args:
            importance (torch.Tensor): Shape (batch, features), from FC1 layer.
            activations (torch.Tensor): Shape (batch, timesteps, features), before max-pooling.

        Returns:
            torch.Tensor: Shape (batch, timesteps, features), importance distributed over time.
        """
        batch_size, timesteps, num_features = activations.shape

        # 🚀 Step 1: Find which timestep contributed most to max-pooling
        max_indices = activations.abs().max(dim=1).indices  # Shape: (batch,)

        # 🚀 Step 2: Create an empty tensor to store the redistributed importance
        spread_importance = torch.zeros_like(activations)  # Shape: (batch, timesteps, features)

        # 🚀 Step 3: Apply Gaussian spreading based on max timestep
        for i in range(batch_size):
            max_t = max_indices[i].item()  # The timestep where max was selected

            # Create Gaussian weights centered at max_t
            time_range = torch.arange(timesteps, device=importance.device)
            gaussian_weights = torch.exp(-((time_range - max_t) ** 2) / (2 * (timesteps / 5) ** 2))  # Adjust sigma

            # Normalize weights so they sum to 1
            gaussian_weights = gaussian_weights / gaussian_weights.sum()

            # Expand to match feature dimension and apply to importance
            spread_importance[i] = gaussian_weights[:, None] * importance[i]  # Shape: (timesteps, features)

        print(
            f"✅ Final importance shape after max-pooling backpropagation: {spread_importance.shape}")  # Expected: (batch, timesteps, features)
        return spread_importance
