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
from Utilities import define_model_and_optim
from PyTorchModelWrapper import PyTorchModelWrapper
from FOVAL_Preprocessor import selected_features

import torch
from torch.cuda.amp import autocast

device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU
torch.backends.cudnn.enabled = False


class FeatureRankingsCreator:
    def __init__(self, hyperparameters, de, subject_list):
        self.outputFolder = None
        self.methods = None
        self.memory_data = None
        self.timing_data = None
        self.subject_list = None
        self.de = None
        self.hyperparameters = None
        self.setup_directories()
        self.feature_importance_results = {}  # Dictionary to store feature importances for each method
        self.initialize_hyperparameters(hyperparameters, de, subject_list)

    def setup_directories(self):
        self.outputFolder = 'FeaturesRankings_Creation'
        os.makedirs(self.outputFolder, exist_ok=True)

    def initialize_hyperparameters(self, hyperparameters, de, subject_list):
        self.hyperparameters = hyperparameters
        self.de = de
        self.subject_list = subject_list
        self.timing_data = []
        self.memory_data = []
        self.methods = [
            'nisp',
            'deeplift',
            'fast_shap_values',
            'captum_intGrad',
            'actif_robust_penHigh',
            'actif_mean',
            'actif_weighted_mean',
            'actif_inverted_weighted_mean',
            'actif_mean_stddev',
            'actif_robust',
            'ablation',
            'shuffle',
            'actif_lin'
        ]

    def process_methods(self):
        for method in self.methods:
            logging.info(f"Evaluating method: {method}")
            aggregated_importances = self.aggregate_feature_importances_for_all_subjects(method=method)
            self.sort_importances_based_on_attribution(aggregated_importances, method=method)

    def aggregate_feature_importances_for_all_subjects(self, method='captum_intGrad'):
        aggregated_importances = []
        all_execution_times = []
        all_memory_usages = []

        for test_subject in self.subject_list:
            logging.info(f"Processing subject: {test_subject}")
            userFolder = f'../results/{test_subject}'

            trained_model, train_loader, valid_loader, input_size = self.prepare_for_feature_importance_visualization_all(
                test_subject)

            method_func = self.get_method_function(method, trained_model, valid_loader, userFolder)
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

    def get_method_function(self, method, trained_model, valid_loader, userFolder):
        method_functions = {
            'actif_robust_penHigh': lambda: self.actif_robust_penHigh(userFolder),
            'actif_mean': lambda: self.actif_mean(userFolder),
            'actif_mean_stddev': lambda: self.actif_mean_stddev(userFolder),
            'actif_weighted_mean': lambda: self.actif_weighted_mean(userFolder),
            'actif_inverted_weighted_mean': lambda: self.actif_inverted_weighted_mean(userFolder),
            'actif_robust': lambda: self.actif_robust(userFolder),
            'shuffle': lambda: self.feature_shuffling_importances(trained_model, valid_loader, userFolder),
            'ablation': lambda: self.ablation(trained_model, valid_loader),
            'captum_intGrad': lambda: self.captum_intGrad(trained_model, valid_loader),
            'fast_shap_values': lambda: self.compute_fast_shap_values(trained_model, valid_loader, device),
            'deeplift': lambda: self.compute_deeplift(trained_model, valid_loader, device),
            'nisp': lambda: self.compute_nisp(trained_model, valid_loader, device)
        }
        return method_functions.get(method, None)

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

    def prepare_for_feature_importance_visualization_all(self, test_subject):
        model = None
        batch_size = self.hyperparameters['batch_size']
        userFolder = f'../results/{test_subject}'
        validation_subjects = test_subject
        remaining_subjects = np.setdiff1d(self.subject_list, validation_subjects)

        logging.info(f"Validation subject(s): {validation_subjects}")
        logging.info(f"Training subjects: {remaining_subjects}")

        train_loader, valid_loader, test_loader, input_size = self.de.get_data_loader(
            remaining_subjects, validation_subjects, None, batch_size=batch_size)

        model, optimizer, scheduler = define_model_and_optim(
            model=None,
            input_size=input_size,
            embed_dim=self.hyperparameters['embed_dim'],
            dropoutRate=self.hyperparameters['dropout_rate'],
            learning_rate=self.hyperparameters['learning_rate'],
            weight_decay=self.hyperparameters['weight_decay'],
            fc1_dim=self.hyperparameters['fc1_dim']
        )

        # model_path = os.path.join(userFolder, 'optimal_subject_model_state_dict.pth')
        # model.load_state_dict(torch.load(model_path))

        return model, train_loader, valid_loader, input_size

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

    # ================ ACTIF METHODS

    def actif_mean(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_mean)

    def actif_mean_stddev(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_meanstddev)

    def actif_weighted_mean(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_weighted_mean)

    def actif_inverted_weighted_mean(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_inverted_weighted_mean)

    def actif_robust(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_robust)

    def actif_robust_penHigh(self, user_folder):
        return self.actif_calculation(user_folder, self.calculate_actif_robust_penHigh)

    def actif_calculation(self, user_folder, calculation_function):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
        all_importances = []
        activation_file_path = os.path.join(user_folder, 'intermediates_activations.npy')

        if os.path.exists(activation_file_path):
            activations_data = np.load(activation_file_path, allow_pickle=True).item()
            layer_activation = list(activations_data.values())[1]

            if isinstance(layer_activation, np.ndarray) and layer_activation.size > 0:
                if layer_activation.ndim > 2:
                    layer_activation = np.mean(layer_activation, axis=1)

                importance, _, _ = calculation_function(layer_activation)
                all_importances.append(importance)

        all_mean_importances = np.mean(np.array(all_importances), axis=0)
        sorted_indices = np.argsort(-all_mean_importances)
        sorted_features = np.array(selected_features2)[sorted_indices]
        sorted_all_mean_importances = all_mean_importances[sorted_indices]

        results = [{'feature': feature, 'attribution': sorted_all_mean_importances[i]} for i, feature in
                   enumerate(sorted_features)]
        return results

    # ================ COMPUTING METHODS

    def calculate_actif_mean(self, activation):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)
        return mean_activation, mean_activation, std_activation

    def calculate_actif_meanstddev(self, activation):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)
        weighted_importance = mean_activation * std_activation
        return weighted_importance, mean_activation, std_activation

    def calculate_actif_weighted_mean(self, activation):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation))
        std_activation = np.std(activation_abs, axis=0)
        normalized_std = (std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation))
        adjusted_importance = (normalized_mean + normalized_std) / 2
        return adjusted_importance, mean_activation, std_activation

    def calculate_actif_inverted_weighted_mean(self, activation):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation))
        std_activation = np.std(activation_abs, axis=0)
        inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (
                np.max(std_activation) - np.min(std_activation))
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance, mean_activation, std_activation

    def calculate_actif_robust(self, activations, epsilon=0.01, min_std_threshold=0.01):
        activation_abs = np.abs(activations)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation) + epsilon)
        transformed_std = np.exp(-std_activation / min_std_threshold)
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

    def feature_shuffling_importances(self, trained_model, valid_loader, userfolder):
        cols = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
        overall_baseline_mae, intermediate_activations, y_batch = self.de.calculateBaseLine(trained_model, valid_loader)

        self.de.visualize_activations(intermediate_activations, y_batch, userfolder=userfolder, name="baseline")
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

    def compute_fast_shap_values(self, model, valid_loader, device):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
        shap_values_accumulated = []
        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)
            background_data = input_batch[:20].detach().cpu().numpy().reshape(-1, 540)
            input_instance = input_batch.detach().cpu().numpy().reshape(input_batch.size(0), -1)

            model_wrapper = PyTorchModelWrapper(model, (10, 54))
            explainer = shap.KernelExplainer(model_wrapper, background_data, link="identity", algorithm="auto")
            shap_values = explainer.shap_values(input_instance, nsamples=300)
            shap_values_accumulated.append(shap_values)

        if len(shap_values_accumulated) == 0:
            logging.warning("No SHAP values accumulated.")
            return None

        shap_values_np = np.concatenate(shap_values_accumulated, axis=1)
        shap_values_reshaped = shap_values_np.reshape(-1, 10, 54)
        mean_shap_values_timesteps = np.mean(shap_values_reshaped, axis=1)
        mean_shap_values_combined = np.mean(mean_shap_values_timesteps, axis=0)

        if mean_shap_values_combined.shape != (54,):
            logging.error(
                f"Incorrect shape of mean_shap_values. Expected (54,), got: {mean_shap_values_combined.shape}")
            return None

        shap_values_df = pd.DataFrame([mean_shap_values_combined], columns=selected_features2)
        results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                   shap_values_df.items()]

        model.eval()
        return results

    def ablation(self, model, valid_loader):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
        model.eval()
        feature_ablation = FeatureAblation(lambda input_batch: self.model_wrapper(model, input_batch))

        aggregated_attributions = torch.zeros(10, len(selected_features2)).to(device)
        instance_count = 0

        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)
            attribution = feature_ablation.attribute(input_batch)
            aggregated_attributions += attribution.sum(dim=0)
            instance_count += input_batch.size(0)

        mean_attributions = aggregated_attributions.cpu().numpy() / instance_count
        mean_attributions = mean_attributions.mean(axis=0)

        results = [{'feature': feature, 'attribution': mean_attributions[i]} for i, feature in
                   enumerate(selected_features2)]
        return results

    def captum_intGrad(self, model, valid_loader):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
        accumulated_attributions = torch.zeros(10, len(selected_features2), device=device)
        total_batches = 0

        model.eval()
        for inputs, _ in valid_loader:
            inputs = inputs.to(device)
            explainer = IntegratedGradients(lambda input_batch: self.model_wrapper(model, input_batch))
            attributions = explainer.attribute(inputs)
            accumulated_attributions += attributions.sum(dim=0)
            total_batches += 1

        if total_batches > 0:
            mean_attributions = accumulated_attributions / total_batches
            mean_attributions = mean_attributions.cpu()

            attributions_df = pd.DataFrame(mean_attributions.numpy(), columns=selected_features2)
            mean_abs_attributions = attributions_df.abs().mean()
            feature_importance = mean_abs_attributions.sort_values(ascending=False)

            results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                       feature_importance.items()]
            return results
        else:
            raise ValueError("No batches processed. Check your data loader and inputs.")

    def model_wrapper(self, model, input_tensor):
        output = model(input_tensor, return_intermediates=False)
        return output.squeeze(-1)

    def compute_deeplift(self, trained_model, valid_loader, device):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]

        if len(valid_loader) == 0:
            print("Validation loader is empty. Skipping subject.")
            return None  # Or log the subject and continue

        accumulated_attributions = torch.zeros(10, len(selected_features2), device=device)
        total_batches = 0

        trained_model.eval()  # Put model in evaluation mode

        for inputs, _ in valid_loader:
            inputs = inputs.to(device)
            explainer = DeepLift(trained_model)  # Initialize DeepLIFT with the model
            attributions = explainer.attribute(inputs, baselines=torch.zeros_like(inputs))  # Use a zero baseline
            accumulated_attributions += attributions.sum(dim=0)
            total_batches += 1

        if total_batches > 0:
            # Use detach() before calling .cpu() to avoid the gradient-tracking error
            mean_attributions = (accumulated_attributions / total_batches).detach().cpu().numpy()

            attributions_df = pd.DataFrame(mean_attributions, columns=selected_features2)
            mean_abs_attributions = attributions_df.abs().mean()
            feature_importance = mean_abs_attributions.sort_values(ascending=False)

            results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                       feature_importance.items()]
            print("Finished DEEPLift")
            return results
        else:
            print("No batches processed for this subject.")
            return None  # Skip this subject and return a meaningful placeholder


    def compute_nisp(self, trained_model, valid_loader, device, accumulation_steps=1, use_mixed_precision=False):
        selected_features2 = [value for value in selected_features if value not in ('SubjectID', 'Gt_Depth')]
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
        importance_scores = torch.zeros(len(selected_features2), device=device)  # Initialize importance scores

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
                reduced_output_importance = output_importance[:, :len(selected_features2)]  # Match feature size

                # Backpropagate importance scores based on activations
                for activation in reversed(activations):
                    if activation.dim() == 3:  # (batch_size, sequence_length, hidden_size)
                        layer_importance = activation.sum(dim=1).mean(dim=0)[:len(selected_features2)]
                    elif activation.dim() == 2:  # (batch_size, hidden_size)
                        layer_importance = activation.mean(dim=0)[:len(selected_features2)]

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
                                  enumerate(selected_features2)]

            return feature_importance
        else:
            print("No batches processed for this subject.")
            return None
