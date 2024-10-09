import os
import timeit
import logging
import torch
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from captum.attr import IntegratedGradients, FeatureAblation, DeepLift
from memory_profiler import memory_usage
from PyTorchModelWrapper import PyTorchModelWrapper

import torch
from torch.cuda.amp import autocast

from models.foval.foval_preprocessor import input_features

device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU
torch.backends.cudnn.enabled = False


class FeatureRankingsCreator:
    def __init__(self, modelName, model, hyperparameters, datasetName, dataset):
        self.outputFolder = None
        self.currentModel = model
        self.currentModelName = modelName
        self.currentDatasetName = datasetName
        self.methods = None
        self.memory_data = None
        self.timing_data = None
        self.selected_features = [value for value in input_features if value not in ('SubjectID', 'Gt_Depth')]
        self.currentDataset = dataset
        self.subject_list = dataset.subject_list
        self.de = None
        self.hyperparameters = hyperparameters
        self.setup_directories()
        self.feature_importance_results = {}  # Dictionary to store feature importances for each method
        self.timing_data = []
        self.memory_data = []
        self.methods = [
            # 'deeplift_zero',
            # 'deeplift_random',
            # 'deeplift_mean',    # ok
            # 'nisp_v1',      # ok
            # 'nisp_v2',      # ok
            # 'nisp_v3',      # ok
            # 'captum_intGrad_v1', # zu wenig speicher
            # 'captum_intGrad_v2', # zu wenig speicher
            # 'captum_intGrad_v3', # zu wenig speicher
            # 'shap_values_v1', #fehler
            # 'shap_values_v2', #fehler
            # 'shap_values_v3', #fehler
            'shuffle',
            'actif_mean',
            'actif_mean_stddev',
            'actif_weighted_mean',
            'actif_inverted_weighted_mean',
            'actif_robust',
            'actif_robust_penHigh',
            'ablation',
            'actif_lin',
        ]

    def setup_directories(self):
        self.outputFolder = f'../results/' + self.currentModelName + '/' + self.currentDatasetName + '/FeaturesRankings_Creation'
        os.makedirs(self.outputFolder, exist_ok=True)

    # 2.
    def process_methods(self):
        for method in self.methods:
            logging.info(f"Evaluating method: {method}")
            aggregated_importances = self.calculate_ranked_list_by_method(method=method)
            self.sort_importances_based_on_attribution(aggregated_importances, method=method)

    # 3.
    def calculate_ranked_list_by_method(self, method='captum_intGrad'):
        aggregated_importances = []
        all_execution_times = []
        all_memory_usages = []

        for test_subject in self.subject_list:
            logging.info(f"Processing subject: {test_subject}")
            userFolder = f'../results/{self.currentModelName}/{self.currentDatasetName}/{test_subject}'

            # create data loaders
            train_loader, valid_loader, input_size = self.getDataLoaders(test_subject)

            method_func = self.get_method_function(method, self.currentModel, valid_loader, userFolder)
            if method_func:
                execution_time, mem_usage, subject_importances = self.calculate_memory_and_execution_time(method_func)
                if subject_importances:
                    all_execution_times.append(execution_time)
                    all_memory_usages.append(max(mem_usage))
                    aggregated_importances.extend(subject_importances)
                trained_model = None  # Clear model

        self.save_timing_data(method, all_execution_times, all_memory_usages)
        df_importances = pd.DataFrame(aggregated_importances)
        self.feature_importance_results[method] = df_importances  # Store the results
        return df_importances

    def getDataLoaders(self, test_subject):
        batch_size = self.hyperparameters['batch_size']
        validation_subjects = test_subject
        remaining_subjects = np.setdiff1d(self.subject_list, validation_subjects)

        logging.info(f"Validation subject(s): {validation_subjects}")
        logging.info(f"Training subjects: {remaining_subjects}")

        train_loader, valid_loader, test_loader, input_size = self.currentDataset.get_data_loader(
            remaining_subjects.tolist(), validation_subjects, None, batch_size=batch_size)

        return train_loader, valid_loader, input_size

    def get_method_function(self, method, trained_model, valid_loader, userFolder):
        method_functions = {
            'actif_robust_penHigh': lambda: self.actif_robust_penHigh(userFolder, valid_loader),
            'actif_mean': lambda: self.actif_mean(userFolder, valid_loader),
            'actif_mean_stddev': lambda: self.actif_mean_stddev(userFolder, valid_loader),
            'actif_weighted_mean': lambda: self.actif_weighted_mean(userFolder, valid_loader),
            'actif_inverted_weighted_mean': lambda: self.actif_inverted_weighted_mean(userFolder, valid_loader),
            'actif_robust': lambda: self.actif_robust(userFolder, valid_loader),
            'actif_lin': lambda: self.actif_lin(userFolder, valid_loader),
            'shuffle': lambda: self.feature_shuffling_importances(trained_model, valid_loader, userFolder),
            'ablation': lambda: self.ablation(trained_model, valid_loader),
            'captum_intGrad_v1': lambda: self.compute_intgrad(trained_model, valid_loader, version='v1'),
            'captum_intGrad_v2': lambda: self.compute_intgrad(trained_model, valid_loader, version='v2'),
            'captum_intGrad_v3': lambda: self.compute_intgrad(trained_model, valid_loader, version='v3'),
            'shap_values_v1': lambda: self.compute_shap(trained_model, valid_loader, device, version='v1'),
            'shap_values_v2': lambda: self.compute_shap(trained_model, valid_loader, device, version='v2'),
            'shap_values_v3': lambda: self.compute_shap(trained_model, valid_loader, device, version='v3'),
            'deeplift_zero': lambda: self.compute_deeplift(trained_model, valid_loader, device, baseline_type='zero'),
            'deeplift_random': lambda: self.compute_deeplift(trained_model, valid_loader, device,
                                                             baseline_type='random'),
            'deeplift_mean': lambda: self.compute_deeplift(trained_model, valid_loader, device, baseline_type='mean'),
            'nisp_v1': lambda: self.compute_nisp(trained_model, valid_loader, device, version='v1'),
            'nisp_v2': lambda: self.compute_nisp(trained_model, valid_loader, device, version='v2'),
            'nisp_v3': lambda: self.compute_nisp(trained_model, valid_loader, device, version='v3')
        }
        return method_functions.get(method, None)

    '''
        ACTIF Variants
    '''
    def actif_mean(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_mean, valid_loader, False)

    def actif_mean_stddev(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_meanstddev, valid_loader, False)

    def actif_weighted_mean(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_weighted_mean, valid_loader, False)

    def actif_inverted_weighted_mean(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_inverted_weighted_mean, valid_loader, False)

    def actif_robust(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_robust, valid_loader, False)

    def actif_robust_penHigh(self, user_folder, valid_loader):
        return self.actif_calculation(user_folder, self.calculate_actif_robust_penHigh, valid_loader, False)

    def actif_calculation(self, user_folder, calculation_function, valid_loader, use_layers):
        all_importances = []

        if use_layers:
            activation_file_path = os.path.join(user_folder, 'intermediates_activations.npy')

            if os.path.exists(activation_file_path):
                activations_data = np.load(activation_file_path, allow_pickle=True).item()
                layer_activation = list(activations_data.values())[1]

                if isinstance(layer_activation, np.ndarray) and layer_activation.size > 0:
                    if layer_activation.ndim > 2:
                        layer_activation = np.mean(layer_activation, axis=1)

                    importance, _, _ = calculation_function(layer_activation)
                    all_importances.append(importance)

        else:
            total_sum = None
            total_count = 0

            # Disable gradient calculation to save memory during inference
            with torch.no_grad():
                # Loop through the validation loader batch by batch
                for batch in valid_loader:
                    inputs, _ = batch  # Assuming batch returns (inputs, labels)
                    inputs = inputs.to(device)  # Move inputs to the appropriate device

                    # inputs shape: [batch_size, time_steps, num_features]
                    # batch_size, time_steps, num_features = inputs.shape

                    # Convert inputs to numpy if needed, and compute mean activations over time
                    inputs_np = inputs.cpu().numpy()  # Convert inputs to numpy array

                    # Compute mean activations over time (axis=1) for each feature
                    mean_activation = np.mean(np.abs(inputs_np), axis=1)  # Mean over time steps

                    # Call the calculation function (e.g., actif_mean) with the mean activations
                    importance, _, _ = calculation_function(mean_activation)
                    all_importances.append(importance)

                    # Clear GPU cache after each batch to prevent memory buildup
                    del inputs
                    torch.cuda.empty_cache()

        # Aggregate all importances
        if all_importances:
            all_mean_importances = np.mean(np.array(all_importances), axis=0)
            sorted_indices = np.argsort(-all_mean_importances)
            sorted_features = np.array(self.selected_features)[sorted_indices]
            sorted_all_mean_importances = all_mean_importances[sorted_indices]

            # Return sorted results as a list of feature importances
            results = [{'feature': feature, 'attribution': sorted_all_mean_importances[i]} for i, feature in
                       enumerate(sorted_features)]
            return results
        else:
            logging.warning("No importance values were calculated.")
            return None

    # ================ COMPUTING METHODS
    def calculate_actif_mean(self, activation):
        """
           Calculate the mean of absolute activations for each feature.

           Args:
               activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

           Returns:
               mean_activation (np.ndarray): The mean activation for each feature.
           """
        activation_abs = np.abs(activation)  # Take the absolute value of activations
        mean_activation = np.mean(activation_abs, axis=0)  # Compute mean across samples
        return mean_activation, mean_activation, None  # Return mean activation as the importance

    def calculate_actif_meanstddev(self, activation):
        """
        Calculate the mean and standard deviation of absolute activations for each feature.

        Args:
            activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

        Returns:
            weighted_importance (np.ndarray): Importance as the product of mean and stddev.
            mean_activation (np.ndarray): Mean of the activations.
            std_activation (np.ndarray): Standard deviation of the activations.
        """
        activation_abs = np.abs(activation)  # Take the absolute value of activations
        mean_activation = np.mean(activation_abs, axis=0)  # Compute mean across samples
        std_activation = np.std(activation_abs, axis=0)  # Compute standard deviation across samples
        weighted_importance = mean_activation * std_activation  # Multiply mean by stddev to get importance
        return weighted_importance, mean_activation, std_activation

    def calculate_actif_weighted_mean(self, activation):
        """
        Calculate a weighted mean of normalized mean activations and standard deviations.

        Args:
            activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

        Returns:
            adjusted_importance (np.ndarray): Weighted importance of normalized mean and stddev.
        """
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)  # Mean over samples
        std_activation = np.std(activation_abs, axis=0)  # Standard deviation over samples

        # Normalize mean and stddev
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation))
        normalized_std = (std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation))

        # Calculate weighted mean of normalized values
        adjusted_importance = (normalized_mean + normalized_std) / 2
        return adjusted_importance, mean_activation, std_activation

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
        return adjusted_importance, mean_activation, std_activation

    def calculate_actif_robust(self, activations, epsilon=0.01, min_std_threshold=0.01):
        """
        Calculate robust importance where features with high mean activations and low variability are preferred.

        Args:
            activations (np.ndarray): The input activations or features.
            epsilon (float): Small value to prevent division by zero.
            min_std_threshold (float): A threshold to control the impact of stddev.

        Returns:
            adjusted_importance (np.ndarray): Robust importance scores.
        """
        activation_abs = np.abs(activations)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        # Normalize the mean and penalize stddev
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation) + epsilon)
        transformed_std = np.exp(-std_activation / min_std_threshold)  # Exponentially penalize high stddev
        adjusted_importance = normalized_mean * (1 - transformed_std)

        return adjusted_importance, mean_activation, std_activation

    def calculate_actif_robust_penHigh(self, activations, epsilon=0.01, min_std_threshold=0.01):
        activation_abs = np.abs(activations)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation) + epsilon)
        transformed_std = np.exp(-std_activation / min_std_threshold)
        adjusted_importance = normalized_mean * transformed_std

        return adjusted_importance, mean_activation, std_activation

    '''
        Ablation
    '''
    def ablation(self, model, valid_loader):
        model.eval()
        feature_ablation = FeatureAblation(lambda input_batch: self.model_wrapper(model, input_batch))

        aggregated_attributions = torch.zeros(10, len(self.selected_features)).to(device)
        instance_count = 0

        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)
            attribution = feature_ablation.attribute(input_batch)
            aggregated_attributions += attribution.sum(dim=0)
            instance_count += input_batch.size(0)

        mean_attributions = aggregated_attributions.cpu().numpy() / instance_count
        mean_attributions = mean_attributions.mean(axis=0)

        results = [{'feature': feature, 'attribution': mean_attributions[i]} for i, feature in
                   enumerate(self.selected_features)]
        return results

    # def captum_intGrad(self, model, valid_loader):
    #     accumulated_attributions = torch.zeros(10, len(self.selected_features), device=device)
    #     total_batches = 0
    #
    #     model.eval()
    #     for inputs, _ in valid_loader:
    #         inputs = inputs.to(device)
    #         explainer = IntegratedGradients(lambda input_batch: self.model_wrapper(model, input_batch))
    #         attributions = explainer.attribute(inputs)
    #         accumulated_attributions += attributions.sum(dim=0)
    #         total_batches += 1
    #
    #     if total_batches > 0:
    #         mean_attributions = accumulated_attributions / total_batches
    #         mean_attributions = mean_attributions.cpu()
    #
    #         attributions_df = pd.DataFrame(mean_attributions.numpy(), columns=self.selected_features)
    #         mean_abs_attributions = attributions_df.abs().mean()
    #         feature_importance = mean_abs_attributions.sort_values(ascending=False)
    #
    #         results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
    #                    feature_importance.items()]
    #         return results
    #     else:
    #         raise ValueError("No batches processed. Check your data loader and inputs.")

    '''
        Deep Lift
    '''
    def compute_deeplift(self, trained_model, valid_loader, device, baseline_type='zero'):
        accumulated_attributions = torch.zeros(10, len(self.selected_features), device=device)
        total_batches = 0

        trained_model.eval()  # Put model in evaluation mode

        for inputs, _ in valid_loader:
            inputs = inputs.to(device)

            # Define the baseline based on the selected baseline_type
            if baseline_type == 'zero':
                baselines = torch.zeros_like(inputs)  # Zero baseline
            elif baseline_type == 'random':
                baselines = torch.rand_like(inputs)  # Random baseline (uniform between 0 and 1)
            elif baseline_type == 'mean':
                # Compute the mean baseline (per feature) along dimension 0
                mean_baseline = torch.mean(inputs, dim=0)

                # Expand the mean_baseline to match the shape of the input
                baselines = mean_baseline.expand_as(inputs)
                # baselines = torch.full_like(inputs, torch.mean(inputs, dim=0).)  # Mean baseline (per feature)
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")

            explainer = DeepLift(trained_model)  # Initialize DeepLIFT with the model
            attributions = explainer.attribute(inputs, baselines=baselines)  # Use the chosen baseline
            accumulated_attributions += attributions.sum(dim=0)
            total_batches += 1

        if total_batches > 0:
            # Calculate the mean attributions across batches
            mean_attributions = (accumulated_attributions / total_batches).detach().cpu().numpy()

            # Store the attributions as a dataframe
            attributions_df = pd.DataFrame(mean_attributions, columns=self.selected_features)
            mean_abs_attributions = attributions_df.abs().mean()
            feature_importance = mean_abs_attributions.sort_values(ascending=False)

            # Return the feature importance results
            results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                       feature_importance.items()]
            return results
        else:
            print("No batches processed.")
            return None

    '''
        NISP
    '''

    def compute_nisp(self, trained_model, valid_loader, device, version='v1'):
        if version == 'v1':
            return self.compute_nisp_configured(trained_model, valid_loader, accumulation_steps=1,
                                                use_mixed_precision=False)
        elif version == 'v2':
            return self.compute_nisp_configured(trained_model, valid_loader, accumulation_steps=1,
                                                use_mixed_precision=True)

        elif version == 'v3':
            return self.compute_nisp_v3(trained_model, valid_loader, device, accumulation_steps=1,
                                        use_mixed_precision=False)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_nisp_v3(self, trained_model, valid_loader, device, accumulation_steps=1, use_mixed_precision=False):
        activations = []

        # Register hook only for the last LSTM layer
        def save_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # Extract the hidden states
                activations.append(output.detach())  # Detach to avoid tracking gradients

            return hook

        # Hook only the last LSTM layer (assuming named 'lstm')
        for name, layer in trained_model.named_modules():
            if isinstance(layer, torch.nn.LSTM) and 'lstm' in name:  # Adjust for your LSTM layer name
                layer.register_forward_hook(save_activation(name))

        trained_model.eval()
        importance_scores = torch.zeros(len(self.selected_features), device=device)

        total_batches = 0

        if len(valid_loader) == 0:
            print("Skipping subject: The validation loader is empty.")
            return None

        for i, (inputs, _) in enumerate(valid_loader):
            inputs = inputs.to(device)

            with autocast(enabled=use_mixed_precision):
                outputs = trained_model(inputs)

                if outputs.dim() == 3:
                    output_importance = outputs.mean(dim=1)
                elif outputs.dim() == 2:
                    output_importance = outputs
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                reduced_output_importance = output_importance[:, :len(self.selected_features)]

                # Backpropagate importance scores based on activations from selected layer
                for activation in reversed(activations):
                    if activation.dim() == 3:
                        layer_importance = activation.sum(dim=1).mean(dim=0)[:len(self.selected_features)]
                    elif activation.dim() == 2:
                        layer_importance = activation.mean(dim=0)[:len(self.selected_features)]
                    importance_scores += layer_importance * reduced_output_importance.mean(dim=0)

            total_batches += 1
            activations.clear()

            torch.cuda.empty_cache()

        if total_batches > 0:
            importance_scores = importance_scores / total_batches
            importance_scores = importance_scores.detach().cpu().numpy()

            feature_importance = [{'feature': feature, 'attribution': importance_scores[i]} for i, feature in
                                  enumerate(self.selected_features)]
            return feature_importance
        else:
            print("No batches processed for this subject.")
            return None

    def compute_nisp_configured(self, trained_model, valid_loader, accumulation_steps=1,
                                use_mixed_precision=False):

        activations = []

        # Register hook to capture activations of each LSTM layer
        def save_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # Extract the hidden states
                activations.append(output.detach())  # Detach to avoid tracking gradients

            return hook

        # Hook into all LSTM layers to capture activations
        for name, layer in trained_model.named_modules():
            if isinstance(layer, torch.nn.LSTM):
                layer.register_forward_hook(save_activation(name))

        trained_model.eval()
        importance_scores = torch.zeros(len(self.selected_features), device=device)  # Initialize importance scores

        total_batches = 0  # For accumulation step counting

        # Check if the valid_loader is empty and skip the subject if it is
        if len(valid_loader) == 0:
            print("Skipping subject: The validation loader is empty.")
            return None

        for i, (inputs, _) in enumerate(valid_loader):
            inputs = inputs.to(device)

            # Mixed precision context
            with autocast(enabled=use_mixed_precision):
                outputs = trained_model(inputs)  # Forward pass to trigger hooks and get activations

                # If outputs have multiple dimensions, handle them appropriately
                if outputs.dim() == 3:  # (batch_size, sequence_length, hidden_size)
                    output_importance = outputs.mean(dim=1)  # Mean over the sequence length
                elif outputs.dim() == 2:  # (batch_size, hidden_size)
                    output_importance = outputs  # Already in correct shape
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                # Ensure size matches the number of input features
                reduced_output_importance = output_importance[:, :len(self.selected_features)]  # Match feature size

                # Backpropagate importance scores based on activations
                for activation in reversed(activations):
                    if activation.dim() == 3:  # (batch_size, sequence_length, hidden_size)
                        layer_importance = activation.sum(dim=1).mean(dim=0)[:len(self.selected_features)]
                    elif activation.dim() == 2:  # (batch_size, hidden_size)
                        layer_importance = activation.mean(dim=0)[:len(self.selected_features)]

                    # Adjust importance scores with reduced output importance
                    importance_scores += layer_importance * reduced_output_importance.mean(dim=0)

            total_batches += 1
            activations.clear()  # Clear activations for the next batch

            # Free GPU cache after each batch
            torch.cuda.empty_cache()

        # Normalize the importance scores by the number of batches
        if total_batches > 0:
            importance_scores = importance_scores / total_batches

            # Ensure the importance scores are moved to CPU and converted to numpy
            importance_scores = importance_scores.detach().cpu().numpy()

            # Create a list of feature importances
            feature_importance = [{'feature': feature, 'attribution': importance_scores[i]} for i, feature in
                                  enumerate(self.selected_features)]

            return feature_importance
        else:
            print("No batches processed for this subject.")
            return None

    '''
    Integrated Gradients
    '''

    def compute_intgrad(self, model, valid_loader, version='v1'):
        if version == 'v1':
            return self.compute_intgrad_configured(model, valid_loader, baseline=torch.zeros_like, steps=100)
        elif version == 'v2':
            return self.compute_intgrad_configured(model, valid_loader, baseline='random', steps=200)
        elif version == 'v3':
            return self.compute_intgrad_configured(model, valid_loader, baseline='mean', steps=50)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_intgrad_configured(self, model, valid_loader, baseline, steps=100):

        accumulated_attributions = torch.zeros(10, len(self.selected_features), device=device)
        total_batches = 0

        model.eval()
        for inputs, _ in valid_loader:
            inputs = inputs.to(device)
            # explainer = IntegratedGradients(model)

            # Calculate attributions
            # attributions = explainer.attribute(inputs, baselines=baseline(inputs), n_steps=steps)
            explainer = IntegratedGradients(lambda input_batch: self.model_wrapper(model, input_batch))
            attributions = explainer.attribute(inputs)

            accumulated_attributions += attributions.sum(dim=0)
            total_batches += 1

        if total_batches > 0:
            mean_attributions = accumulated_attributions / total_batches
            mean_attributions = mean_attributions.cpu().numpy()

            results = [{'feature': feature, 'attribution': mean_attributions[i]} for i, feature in
                       enumerate(self.selected_features)]
            return results
        else:
            raise ValueError("No batches processed. Check your data loader and inputs.")

    '''
        SHAP Values
    '''

    def compute_shap(self, model, valid_loader, device, version='v1'):
        # Define the baseline based on the selected baseline_type
        if version == 'v1':
            return self.compute_shap_configured(model, valid_loader, background_size=10, nsamples=100)
        elif version == 'v2':
            return self.compute_shap_configured(model, valid_loader, background_size=50, nsamples=300)
        elif version == 'v3':
            return self.compute_shap_configured(model, valid_loader, background_kmeans=True, nsamples=500)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_shap_configured(self, model, valid_loader, background_size=20, nsamples=100, background_kmeans=False):
        shap_values_accumulated = []

        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)

            # Background data sampling
            if background_kmeans:
                background_data = shap.kmeans(input_batch.cpu().numpy(), background_size).data
            else:
                background_data = input_batch[:background_size].cpu().numpy()

            input_instance = input_batch.cpu().numpy().reshape(input_batch.size(0), -1)
            model_wrapper = PyTorchModelWrapper(model, (10, 38))

            explainer = shap.KernelExplainer(model_wrapper, background_data)
            shap_values = explainer.shap_values(input_instance, nsamples=nsamples)
            shap_values_accumulated.append(shap_values)

        shap_values_np = np.concatenate(shap_values_accumulated, axis=1)
        shap_values_reshaped = shap_values_np.reshape(-1, 10, 38)
        mean_shap_values_timesteps = np.mean(shap_values_reshaped, axis=1)
        mean_shap_values_combined = np.mean(mean_shap_values_timesteps, axis=0)

        if mean_shap_values_combined.shape != (38,):
            raise ValueError(f"Expected (38,), got: {mean_shap_values_combined.shape}")

        shap_values_df = pd.DataFrame([mean_shap_values_combined], columns=self.selected_features)
        results = [{'feature': feature, 'attribution': attribution} for feature, attribution in shap_values_df.items()]
        return results

    '''
        ACTIF HL
    '''

    def actif_lin(self, userFolder, valid_loader):
        pass

    def model_wrapper(self, model, input_tensor):
        output = model(input_tensor, return_intermediates=False)
        return output.squeeze(-1)

    def feature_shuffling_importances(self, trained_model, valid_loader, userfolder):
        cols = self.selected_features
        overall_baseline_mae, y_batch = self.calculateBaseLine(trained_model, valid_loader)

        # self.de.visualize_activations(intermediate_activations, y_batch, userfolder=userfolder, name="baseline")
        results = [{'feature': 'BASELINE', 'attribution': overall_baseline_mae}]
        trained_model.eval()
        for k in tqdm(range(len(cols)), desc="Computing feature importance"):
            all_attributions_of_feature_k_as_mae = []
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch_shuffled = X_batch.clone()
                indices = torch.randperm(X_batch.size(0))
                X_batch_shuffled[:, :, k] = X_batch_shuffled[indices, :, k]

                with torch.no_grad():
                    oof_preds_shuffled = trained_model(X_batch_shuffled, return_intermediates=False).squeeze()
                    attribution_as_mae = torch.mean(torch.abs(oof_preds_shuffled - y_batch)).item()
                all_attributions_of_feature_k_as_mae.append(attribution_as_mae)

            all_batches_feature_k_mean_attributions = np.mean(all_attributions_of_feature_k_as_mae)
            results.append({'feature': cols[k], 'attribution': all_batches_feature_k_mean_attributions})

        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='attribution', ascending=False)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        bar_colors = ['#376B74' if x > 0 else '#993768' for x in df_sorted['attribution']]
        plt.barh(df_sorted['feature'], df_sorted['attribution'], color=bar_colors)
        plt.xlabel('Change in MAE (Importance)')
        plt.title('Feature Importance (By Permutation Impact)')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig(f"{userfolder}/FeatureShuffling_Sorted.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        return results


    def calculate_memory_and_execution_time(self, method_func):
        start_time = timeit.default_timer()
        mem_usage, subject_importances = memory_usage((method_func,), retval=True, interval=0.1, timeout=None)
        execution_time = timeit.default_timer() - start_time
        logging.info(f"Execution time: {execution_time} seconds")
        logging.info(f"Memory usage: {max(mem_usage)} MiB")
        return execution_time, mem_usage, subject_importances

    def save_timing_data(self, method, all_execution_times, all_memory_usages):
        if all_execution_times:
            average_time = sum(all_execution_times) / len(all_execution_times)
            total_time = sum(all_execution_times)
            average_memory = sum(all_memory_usages) / len(all_memory_usages)
            total_memory = sum(all_memory_usages)
            self.timing_data.append({
                'Method': method,
                'Average Execution Time': average_time,
                'Total Execution Time': total_time,
                'Average Memory Usage': average_memory,
                'Total Memory Usage': total_memory
            })
            logging.info(f"Average execution time for {method}: {average_time} seconds")
            logging.info(f"Average memory usage for {method}: {average_memory} MiB")

        df_timing = pd.DataFrame(self.timing_data)
        file_path = f"{self.outputFolder}/method_execution_times.csv"
        header = not os.path.exists(file_path)
        df_timing.to_csv(file_path, mode='a', index=False, header=header)
        logging.info(f"Appended average execution times to '{file_path}'")

    def sort_importances_based_on_attribution(self, df_importances, method=None):
        logging.info(f"Sorting importances for method {method}")
        mean_importances = df_importances.groupby('feature')['attribution'].mean().reset_index()
        mean_importances_sorted = mean_importances.sort_values(by='attribution', ascending=False)

        plt.figure(figsize=(10, 15))
        bars = plt.barh(mean_importances_sorted['feature'], mean_importances_sorted['attribution'], color='skyblue')
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.2f}", va='center', rotation=90)
        plt.xlabel('attribution')
        plt.title('Feature Importance Across All Subjects')
        plt.tight_layout()
        plt.savefig(f"{self.outputFolder}/{method}_importances.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        mean_importances = df_importances.groupby('feature')['attribution'].mean().reset_index()
        mean_importances_sorted = mean_importances.sort_values(by='attribution', ascending=False)
        self.feature_importance_results[method] = mean_importances_sorted['feature'].tolist()  # Store sorted features
        self.save_importances_in_file(mean_importances_sorted, method)

    def save_importances_in_file(self, mean_importances_sorted, method):
        filename = f"{self.outputFolder}/{method}.csv"
        mean_importances_sorted.to_csv(filename, index=False)
        logging.info(f"Saved importances for {method} in {filename}")

    import torch

    def calculateBaseLine(self, trained_model, valid_loader):
        """
        Calculate the baseline MAE for the model on the validation dataset.
        This is done without shuffling any features, serving as a reference point for feature importance.
        """
        trained_model.eval()  # Set the model to evaluation mode
        all_baseline_maes = []  # To store MAE for each batch
        all_y_batches = []  # To store all true labels for reference

        # Iterate over the validation data loader
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Make predictions without shuffling
            with torch.no_grad():
                oof_preds = trained_model(X_batch, return_intermediates=False).squeeze()

            # Calculate MAE for the current batch
            batch_mae = torch.mean(torch.abs(oof_preds - y_batch)).item()
            all_baseline_maes.append(batch_mae)

            # Store the true labels for reference (optional, depending on your use case)
            all_y_batches.append(y_batch.cpu().numpy())

        # Calculate the overall mean baseline MAE across all batches
        overall_baseline_mae = np.mean(all_baseline_maes)

        # Combine all true labels if you need them later for any purpose
        all_y_batches = np.concatenate(all_y_batches, axis=0)

        return overall_baseline_mae, all_y_batches
