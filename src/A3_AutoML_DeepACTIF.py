import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from src.models.FOVAL.FOVAL import MaxOverTimePooling


class A3_AutoML_DeepACTIF:
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
                # if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                #     _, max_idx = torch.max(output, dim=1)  # Shape (batch, features)
                #     self.max_indices[name] = max_idx  # 🔹 Save indices by layer name

                if isinstance(module, MaxOverTimePooling):
                    _, max_idx = torch.max(output, dim=1)
                    self.max_indices[name] = max_idx
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

    def forward_pass_with_importance(self, inputs, return_raw=False):
        """
        Führt einen Vorwärtsdurchlauf durch und berechnet dabei Importances.

        Args:
            inputs: Eingabetensor der Form (batch, timesteps, ...).
            return_raw (bool): Wenn True, wird der Tensor mit Zeitinformation (Form: [batch, timesteps, num_features])
                               zurückgegeben, andernfalls wird über die Zeit (z. B. Mittelwert über die Zeit) aggregiert.

        Returns:
            Tensor mit rohen Importances (batch, timesteps, len(selected_features))
            bzw. aggregiert (batch, len(selected_features)).
        """
        self.model.eval()
        activations = inputs
        # Bestimme T als ursprüngliche Zeitdimension
        T = inputs.shape[1]
        accumulated_importance = None

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.LSTM):
                activations, _ = layer(activations)  # Ergebnis: (batch, L, hidden)
                layer_importance = torch.abs(activations)  # (batch, L, hidden)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)  # Erwartete Form: entweder (batch, features) oder (batch, T, features)
                if activations.dim() == 2:
                    # Wenn 2D, erweitern wir ihn auf 3D: (batch, 1, features) -> expand auf (batch, T, features)
                    layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
                else:
                    # Falls schon 3D (batch, T, features), direkt verwenden:
                    layer_importance = torch.abs(activations)
            else:
                continue  # Andere Schichten überspringen

            # --- Interpolation entlang der Feature-Dimension ---
            # Ziel: Ändere die letzte Dimension von "orig_features" auf len(selected_features),
            # dabei bleibt die Zeit-Dimension unverändert.
            # Dafür formen wir zuerst um:
            N, T_current, orig_features = layer_importance.shape
            # Umformen zu (N*T, 1, orig_features)
            reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            # Interpolieren mit Linear-Interpolation (1D) auf die gewünschte Länge
            interpolated = F.interpolate(reshaped, size=len(self.selected_features),
                                         mode="linear", align_corners=False)
            # Zurückformen zu (N, T, len(selected_features))
            interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
            # --- Ende Interpolation ---

           #  print(f"Layer {name} -> interpolated_importance shape: {interpolated_importance.shape}")

            # Akkumulieren: Falls erster Layer, direkt setzen, sonst addieren.
            if accumulated_importance is None:
                accumulated_importance = interpolated_importance
            else:
                if accumulated_importance.shape != interpolated_importance.shape:
                    raise ValueError(
                        f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}")
                accumulated_importance += interpolated_importance

        # Falls raw gewünscht: Rückgabe des Tensor mit Zeitinformation
        if return_raw:
            return accumulated_importance
        else:
            # Aggregiere über die Zeit (z. B. Mittelwert über dim=1)
            aggregated_importance = accumulated_importance.mean(dim=1)
            return aggregated_importance

    def analyze(self, dataloader, device, return_raw=False):
        """
        Berechnet Feature-Importances über den gesamten Datensatz.

        Args:
            dataloader: DataLoader für die Samples.
            device: Device, z. B. "cuda" oder "cpu".
            return_raw (bool): Falls True, werden raw Importances mit Zeitinformation zurückgegeben.

        Returns:
            Wenn return_raw=False: Array der Form (num_samples, num_features) mit aggregierten Werten.
            Falls True: Tensor der Form (num_samples, timesteps, num_features).
        """
        all_importances = []

        for batch_idx, batch in enumerate(dataloader):
            # Falls der Batch als (inputs, labels) vorliegt:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, dtype=torch.float32)
            importance = self.forward_pass_with_importance(batch, return_raw=return_raw)

            if importance is None or torch.isnan(importance).any():
                print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
                continue

            all_importances.append(importance)

        if not all_importances:
            raise ValueError("❌ No valid feature importances calculated!")

        # Falls return_raw True, erhalte ein Tensor mit der Form (num_samples, timesteps, num_features)
        if return_raw:
            all_importances = torch.cat(all_importances, dim=0)
            # print("ANAlyzer: ", all_importances)
            return all_importances
        else:
            # Andernfalls aggregiere über den Batch
            all_importances = torch.cat(all_importances, dim=0)  # (num_samples, num_features)
            return all_importances

    def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
        """
        Aggregiert raw Importances (über die Zeit) mittels invers gewichteter Methode pro Sample.

        Args:
            activation (np.ndarray): Array der Form (num_samples, timesteps, num_features).
            epsilon (float): Kleine Konstante zur Stabilisierung.

        Returns:
            np.ndarray: Aggregierte Importance-Werte mit Form (num_samples, num_features).
        """
        # Berechne pro Sample und pro Feature den Mittelwert und die Standardabweichung über die Zeit (axis=1)
        mean_activation = np.mean(np.abs(activation), axis=1)  # (num_samples, num_features)
        std_activation = np.std(np.abs(activation), axis=1)  # (num_samples, num_features)

        # Normalisiere pro Sample (über die Feature-Dimension)
        min_mean = np.min(mean_activation, axis=1, keepdims=True)
        max_mean = np.max(mean_activation, axis=1, keepdims=True)
        normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)

        min_std = np.min(std_activation, axis=1, keepdims=True)
        max_std = np.max(std_activation, axis=1, keepdims=True)
        normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
        inverse_normalized_std = 1 - normalized_std

        aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
        return aggregated_importance

    # Hilfsfunktionen für raw Ergebnisse:
    def most_important_timestep(self, all_attributions):
        """
        Ermittelt pro Sample und pro Feature den Timestep mit maximalem absoluten Wert.

        Args:
            all_attributions (np.ndarray): Array mit Form (num_samples, timesteps, num_features).

        Returns:
            np.ndarray: Array mit Form (num_samples, num_features) mit den Indizes des wichtigsten Timesteps.
        """
        return np.argmax(np.abs(all_attributions), axis=1)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
        """
        Aggregiert raw Importances über die Zeit mittels invers gewichteter Methode.

        Args:
            all_attributions (np.ndarray): Array der Form (num_samples, timesteps, num_features).

        Returns:
            np.ndarray: Aggregierte Importances (num_samples, num_features).
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
        Aggregates sample importances to produce a global importance per feature.

        Args:
            sample_importances (np.ndarray): Array of shape (num_samples, num_features)
                (zum Beispiel aggregierte Importances, in denen pro Sample bereits über die Timesteps gemittelt wurde).

        Returns:
            np.ndarray: Global aggregated importance per feature (shape: (num_features,)).
        """
        # Berechne den Mittelwert über die Samples (axis=0)
        global_importance = np.mean(sample_importances, axis=0)
        return global_importance

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

        print("All Attributions: ", all_attributions)

        # we should return samples, timesteps, features

        # next: mean over timesteps so we have samples,features
        # Aggregiere mithilfe deiner ACTIF-Methode (hier: calculate_actif_inverted_weighted_mean)
        # Hinweis: Falls du pro Sample eine Aggregation möchtest, muss die Dimension stimmen.
        # Ein einfaches Beispiel: Wir nehmen an, all_attributions hat bereits die Form (num_samples, num_features)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)

        # Erstelle das Ergebnis im SHAP-ähnlichen Format: Eine Liste von Dicts pro Sample.
        results = [{'feature': self.selected_features[i], 'attribution': all_attributions[i]} for i in
                   range(len(self.selected_features))]


        print("Results from Class: ", results)


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

