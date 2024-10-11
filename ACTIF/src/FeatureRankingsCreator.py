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
import json
from models.foval.FOVAL import FOVAL

device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU
torch.backends.cudnn.enabled = False


class FeatureRankingsCreator:
    def __init__(self, modelName, datasetName, dataset):
        self.baseFolder = None
        self.outputFolder = None
        self.currentModel = None
        self.currentModelName = modelName
        self.currentDatasetName = datasetName
        self.methods = None
        self.memory_data = None
        self.timing_data = None
        self.selected_features = [value for value in input_features if value not in ('SubjectID', 'Gt_Depth')]
        self.currentDataset = dataset
        self.subject_list = dataset.subject_list
        self.de = None
        self.hyperparameters = None
        self.setup_directories()
        self.feature_importance_results = {}  # Dictionary to store feature importances for each method
        self.timing_data = []
        self.memory_data = []
        self.methods = [
            'actif_mean',                     # model-agnostic: ok
            'actif_mean_stddev',              # ok
            'actif_inverted_weighted_mean',   # ok
            'actif_robust',                   # ok

            'shuffle_MEAN',    # ok
            'shuffle_MEANSTD', # ok
            'shuffle_INV',     # ok
            'shuffle_PEN',     # ok

            'ablation_MEAN',      # ok
            'ablation_MEANSTD',   # ok
            'ablation_INV',       # ok
            'ablation_PEN',       # ok

            'deeplift_zero_MEAN',     # ok
            'deeplift_zero_MEANSTD',  # ok
            'deeplift_zero_INV',      # ok
            'deeplift_zero_PEN',      # ok

            'deeplift_random_MEAN',       # ok
            'deeplift_random_MEANSTD',    # ok
            'deeplift_random_INV',        # ok
            'deeplift_random_PEN',        # ok

            'deeplift_mean_MEAN',         # ok
            'deeplift_mean_MEANSTD',      # ok
            'deeplift_mean_INV',          # ok
            'deeplift_mean_PEN',          # ok

            # 'nisp_v1_MEAN',           no actif variants implemented
            # 'nisp_v1_MEANSTD',
            # 'nisp_v1_INV',
            # 'nisp_v1_PEN',
            #
            # 'nisp_v2_MEAN',
            # 'nisp_v2_MEANSTD',
            # 'nisp_v2_INV',
            # 'nisp_v2_PEN',
            #
            # 'nisp_v3_MEAN',
            # 'nisp_v3_MEANSTD',
            # 'nisp_v3_INV',
            # 'nisp_v3_PEN',

            # 'captum_intGrad_v1_MEAN',         # memory problems + no actif variants implemented
            # 'captum_intGrad_v1_MEANSTD',
            # 'captum_intGrad_v1_INV',
            # 'captum_intGrad_v1_PEN',
            #
            # 'captum_intGrad_v2_MEAN',
            # 'captum_intGrad_v2_MEANSTD',
            # 'captum_intGrad_v2_INV',
            # 'captum_intGrad_v2_PEN',
            #
            # 'captum_intGrad_v3_MEAN',
            # 'captum_intGrad_v3_MEANSTD',
            # 'captum_intGrad_v3_INV',
            # 'captum_intGrad_v3_PEN',

            # 'shap_values_v1_MEAN',            # size mismatch + no actif variants implemented
            # 'shap_values_v1_MEANSTD',
            # 'shap_values_v1_INV',
            # 'shap_values_v1_PEN',
            #
            # 'shap_values_v2_MEAN',
            # 'shap_values_v2_MEANSTD',
            # 'shap_values_v2_INV',
            # 'shap_values_v2_PEN',
            #
            # 'shap_values_v3_MEAN',
            # 'shap_values_v3_MEANSTD',
            # 'shap_values_v3_INV',
            # 'shap_values_v3_PEN',
        ]

    def load_model(self, modelName):
        if modelName == "Foval":
            self.currentModel, self.hyperparameters = self.loadFOVALModel(model_path="../models/foval/config/foval")

    def setup_directories(self):
        self.outputFolder = f'../results/' + self.currentModelName + '/' + self.currentDatasetName + '/FeaturesRankings_Creation'
        os.makedirs(self.outputFolder, exist_ok=True)
        self.baseFolder = os.path.dirname(self.outputFolder)  # This will give the path without

    # 2.
    def process_methods(self):
        for method in self.methods:
            print(f"Evaluating method: {method}")
            aggregated_importances = self.calculate_ranked_list_by_method(method=method)
            self.sort_importances_based_on_attribution(aggregated_importances, method=method)

    # 3.
    def calculate_ranked_list_by_method(self, method='captum_intGrad'):
        aggregated_importances = []
        all_execution_times = []
        all_memory_usages = []

        for i, test_subject in enumerate(self.subject_list):
            print(f"Processing subject {i + 1}/{len(self.subject_list)}: {test_subject}")

            # create data loaders
            train_loader, valid_loader, input_size = self.getDataLoaders(test_subject)

            method_func = self.get_method_function(method, valid_loader)
            if method_func:
                execution_time, mem_usage, subject_importances = self.calculate_memory_and_execution_time(method_func)
                if subject_importances is not None and len(subject_importances) > 0:
                    all_execution_times.append(execution_time)
                    all_memory_usages.append(max(mem_usage))
                    aggregated_importances.extend(subject_importances)

        self.save_timing_data(method, all_execution_times, all_memory_usages)
        df_importances = pd.DataFrame(aggregated_importances)
        self.feature_importance_results[method] = df_importances  # Store the results
        return df_importances

    def getDataLoaders(self, test_subject):
        batch_size = 460
        validation_subjects = test_subject
        remaining_subjects = np.setdiff1d(self.subject_list, validation_subjects)

        logging.info(f"Validation subject(s): {validation_subjects}")
        logging.info(f"Training subjects: {remaining_subjects}")

        train_loader, valid_loader, test_loader, input_size = self.currentDataset.get_data_loader(
            remaining_subjects.tolist(), validation_subjects, None, batch_size=batch_size)

        return train_loader, valid_loader, input_size

    def get_method_function(self, method, valid_loader, load_model=False):
        method_functions = {
            # Actif on Input Methods
            'actif_mean': lambda: self.actif_mean(valid_loader),
            'actif_mean_stddev': lambda: self.actif_mean_stddev(valid_loader),
            'actif_inverted_weighted_mean': lambda: self.actif_inverted_weighted_mean(valid_loader),
            'actif_robust': lambda: self.actif_robust(valid_loader),

            # Shuffle Methods
            'shuffle_MEAN': lambda: self.feature_shuffling_importances(valid_loader, actif_variant='mean'),
            'shuffle_MEANSTD': lambda: self.feature_shuffling_importances(valid_loader, actif_variant='meanstd'),
            'shuffle_INV': lambda: self.feature_shuffling_importances(valid_loader, actif_variant='inv'),
            'shuffle_PEN': lambda: self.feature_shuffling_importances(valid_loader, actif_variant='robust'),

            # Ablation Methods
            'ablation_MEAN': lambda: self.ablation(valid_loader, actif_variant='mean'),
            'ablation_MEANSTD': lambda: self.ablation(valid_loader, actif_variant='meanstd'),
            'ablation_INV': lambda: self.ablation(valid_loader, actif_variant='inv'),
            'ablation_PEN': lambda: self.ablation(valid_loader, actif_variant='robust'),

            # Captum Integrated Gradients Methods (v1, v2, v3)
            'captum_intGrad_v1_MEAN': lambda: self.compute_intgrad(valid_loader, version='v1', actif_variant='mean'),
            'captum_intGrad_v1_MEANSTD': lambda: self.compute_intgrad(valid_loader, version='v1',
                                                                      actif_variant='meanstd'),
            'captum_intGrad_v1_INV': lambda: self.compute_intgrad(valid_loader, version='v1', actif_variant='inv'),
            'captum_intGrad_v1_PEN': lambda: self.compute_intgrad(valid_loader, version='v1', actif_variant='robust'),

            'captum_intGrad_v2_MEAN': lambda: self.compute_intgrad(valid_loader, version='v2', actif_variant='mean'),
            'captum_intGrad_v2_MEANSTD': lambda: self.compute_intgrad(valid_loader, version='v2',
                                                                      actif_variant='meanstd'),
            'captum_intGrad_v2_INV': lambda: self.compute_intgrad(valid_loader, version='v2', actif_variant='inv'),
            'captum_intGrad_v2_PEN': lambda: self.compute_intgrad(valid_loader, version='v2', actif_variant='robust'),

            'captum_intGrad_v3_MEAN': lambda: self.compute_intgrad(valid_loader, version='v3', actif_variant='mean'),
            'captum_intGrad_v3_MEANSTD': lambda: self.compute_intgrad(valid_loader, version='v3',
                                                                      actif_variant='meanstd'),
            'captum_intGrad_v3_INV': lambda: self.compute_intgrad(valid_loader, version='v3', actif_variant='inv'),
            'captum_intGrad_v3_PEN': lambda: self.compute_intgrad(valid_loader, version='v3', actif_variant='robust'),

            # SHAP Values Methods (v1, v2, v3)
            'shap_values_v1_MEAN': lambda: self.compute_shap(valid_loader, version='v1', actif_variant='mean'),
            'shap_values_v1_MEANSTD': lambda: self.compute_shap(valid_loader, version='v1', actif_variant='meanstd'),
            'shap_values_v1_INV': lambda: self.compute_shap(valid_loader, version='v1', actif_variant='inv'),
            'shap_values_v1_PEN': lambda: self.compute_shap(valid_loader, version='v1', actif_variant='robust'),

            'shap_values_v2_MEAN': lambda: self.compute_shap(valid_loader, version='v2', actif_variant='mean'),
            'shap_values_v2_MEANSTD': lambda: self.compute_shap(valid_loader, version='v2', actif_variant='meanstd'),
            'shap_values_v2_INV': lambda: self.compute_shap(valid_loader, version='v2', actif_variant='inv'),
            'shap_values_v2_PEN': lambda: self.compute_shap(valid_loader, version='v2', actif_variant='robust'),

            'shap_values_v3_MEAN': lambda: self.compute_shap(valid_loader, version='v3', actif_variant='mean'),
            'shap_values_v3_MEANSTD': lambda: self.compute_shap(valid_loader, version='v3', actif_variant='meanstd'),
            'shap_values_v3_INV': lambda: self.compute_shap(valid_loader, version='v3', actif_variant='inv'),
            'shap_values_v3_PEN': lambda: self.compute_shap(valid_loader, version='v3', actif_variant='robust'),

            # DeepLIFT Methods with Zero, Random, and Mean Baseline Types
            'deeplift_zero_MEAN': lambda: self.compute_deeplift(valid_loader, baseline_type='zero',
                                                                actif_variant='mean'),
            'deeplift_zero_MEANSTD': lambda: self.compute_deeplift(valid_loader, baseline_type='zero',
                                                                   actif_variant='meanstd'),
            'deeplift_zero_INV': lambda: self.compute_deeplift(valid_loader, baseline_type='zero', actif_variant='inv'),
            'deeplift_zero_PEN': lambda: self.compute_deeplift(valid_loader, baseline_type='zero',
                                                               actif_variant='robust'),

            'deeplift_random_MEAN': lambda: self.compute_deeplift(valid_loader, baseline_type='random',
                                                                  actif_variant='mean'),
            'deeplift_random_MEANSTD': lambda: self.compute_deeplift(valid_loader, baseline_type='random',
                                                                     actif_variant='meanstd'),
            'deeplift_random_INV': lambda: self.compute_deeplift(valid_loader, baseline_type='random',
                                                                 actif_variant='inv'),
            'deeplift_random_PEN': lambda: self.compute_deeplift(valid_loader, baseline_type='random',
                                                                 actif_variant='robust'),

            'deeplift_mean_MEAN': lambda: self.compute_deeplift(valid_loader, baseline_type='mean',
                                                                actif_variant='mean'),
            'deeplift_mean_MEANSTD': lambda: self.compute_deeplift(valid_loader, baseline_type='mean',
                                                                   actif_variant='meanstd'),
            'deeplift_mean_INV': lambda: self.compute_deeplift(valid_loader, baseline_type='mean',
                                                               actif_variant='inv'),
            'deeplift_mean_PEN': lambda: self.compute_deeplift(valid_loader, baseline_type='mean',
                                                               actif_variant='robust'),

            # NISP Methods (v1, v2, v3)
            'nisp_v1_MEAN': lambda: self.compute_nisp(valid_loader, version='v1', actif_variant='mean'),
            'nisp_v1_MEANSTD': lambda: self.compute_nisp(valid_loader, version='v1', actif_variant='meanstd'),
            'nisp_v1_INV': lambda: self.compute_nisp(valid_loader, version='v1', actif_variant='inv'),
            'nisp_v1_PEN': lambda: self.compute_nisp(valid_loader, version='v1', actif_variant='robust'),

            'nisp_v2_MEAN': lambda: self.compute_nisp(valid_loader, version='v2', actif_variant='mean'),
            'nisp_v2_MEANSTD': lambda: self.compute_nisp(valid_loader, version='v2', actif_variant='meanstd'),
            'nisp_v2_INV': lambda: self.compute_nisp(valid_loader, version='v2', actif_variant='inv'),
            'nisp_v2_PEN': lambda: self.compute_nisp(valid_loader, version='v2', actif_variant='robust'),

            'nisp_v3_MEAN': lambda: self.compute_nisp(valid_loader, version='v3', actif_variant='mean'),
            'nisp_v3_MEANSTD': lambda: self.compute_nisp(valid_loader, version='v3', actif_variant='meanstd'),
            'nisp_v3_INV': lambda: self.compute_nisp(valid_loader, version='v3', actif_variant='inv'),
            'nisp_v3_PEN': lambda: self.compute_nisp(valid_loader, version='v3', actif_variant='robust'),
        }

        return method_functions.get(method, None)

    '''
    =======================================================================================
    # ACTIF Variants
    =======================================================================================
    '''

    def actif_mean(self, valid_loader):
        return self.actif_calculation(self.calculate_actif_mean, valid_loader, False)

    def actif_mean_stddev(self, valid_loader):
        return self.actif_calculation(self.calculate_actif_meanstddev, valid_loader, False)

    def actif_weighted_mean(self, valid_loader):
        return self.actif_calculation(self.calculate_actif_weighted_mean, valid_loader, False)

    def actif_inverted_weighted_mean(self, valid_loader):
        return self.actif_calculation(self.calculate_actif_inverted_weighted_mean, valid_loader, False)

    def actif_robust(self, valid_loader):
        return self.actif_calculation(self.calculate_actif_robust, valid_loader, False)

    def actif_calculation(self, calculation_function, valid_loader, use_layers):
        all_importances = []
        total_sum = None
        total_count = 0

        # Disable gradient calculation to save memory during inference
        with torch.no_grad():
            # Loop through the validation loader batch by batch
            for batch in valid_loader:
                inputs, _ = batch  # Assuming batch returns (inputs, labels)
                inputs = inputs.to(device)  # Move inputs to the appropriate device

                # Convert inputs to numpy if needed, and compute mean activations over time
                inputs_np = inputs.cpu().numpy()  # Convert inputs to numpy array

                # Compute mean activations over time (axis=1) for each feature
                mean_activation = np.mean(np.abs(inputs_np), axis=1)  # Mean over time steps

                # Call the calculation function (e.g., actif_mean) with the mean activations
                importance = calculation_function(mean_activation)
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
        return mean_activation  # Return mean activation as the importance

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
        return weighted_importance

    # def calculate_actif_weighted_mean(self, activation):
    #     """
    #     Calculate a weighted mean of normalized mean activations and standard deviations.
    #
    #     Args:
    #         activation (np.ndarray): The input activations or features, shape (num_samples, num_features).
    #
    #     Returns:
    #         adjusted_importance (np.ndarray): Weighted importance of normalized mean and stddev.
    #     """
    #     activation_abs = np.abs(activation)
    #     mean_activation = np.mean(activation_abs, axis=0)  # Mean over samples
    #     std_activation = np.std(activation_abs, axis=0)  # Standard deviation over samples
    #
    #     # Normalize mean and stddev
    #     normalized_mean = (mean_activation - np.min(mean_activation)) / (
    #             np.max(mean_activation) - np.min(mean_activation))
    #     normalized_std = (std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation))
    #
    #     # Calculate weighted mean of normalized values
    #     adjusted_importance = (normalized_mean + normalized_std) / 2
    #     return adjusted_importance, mean_activation, std_activation

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

        return adjusted_importance

    # def calculate_actif_robust_penHigh(self, activations, epsilon=0.01, min_std_threshold=0.01):
    #     activation_abs = np.abs(activations)
    #     mean_activation = np.mean(activation_abs, axis=0)
    #     std_activation = np.std(activation_abs, axis=0)
    #
    #     normalized_mean = (mean_activation - np.min(mean_activation)) / (
    #             np.max(mean_activation) - np.min(mean_activation) + epsilon)
    #     transformed_std = np.exp(-std_activation / min_std_threshold)
    #     adjusted_importance = normalized_mean * transformed_std
    #
    #     return adjusted_importance, mean_activation, std_activation

    '''
    =======================================================================================
    # Established Methods
    =======================================================================================
    '''

    '''
        Ablation
    '''

    def ablation(self, valid_loader, actif_variant='mean'):

        self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")

        self.currentModel.eval()

        total_samples = len(valid_loader.dataset)  # Total number of samples
        aggregated_attributions = torch.zeros(total_samples, len(self.selected_features), device=device)

        start_idx = 0  # To keep track of where to insert the current batch's results

        # Perform ablation
        feature_ablation = FeatureAblation(lambda input_batch: self.model_wrapper(self.currentModel, input_batch))

        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)
            batch_size = input_batch.size(0)

            # Compute feature attributions for the current batch
            attributions = feature_ablation.attribute(input_batch)

            # Aggregate the attributions over the samples in the current batch
            attributions_mean = attributions.mean(dim=1)  # Aggregating over time dimension if necessary

            # Insert the results into the pre-allocated tensor
            aggregated_attributions[start_idx:start_idx + batch_size] = attributions_mean
            start_idx += batch_size

        # Average attributions over all samples
        if start_idx > 0:
            # aggregated_attributions /= total_instances
            # aggregated_attributions = aggregated_attributions.permute(1, 0)  # Shape will now be [batch_size, features, time_steps]

            # # Now, to match ACTIF format, we expand these attributions across the time steps
            # expanded_attributions = aggregated_attributions.repeat(input_batch.size(1), 1)  # Repeat for time steps

            # Call ACTIF variant (e.g., actif_mean) to aggregate the results
            if actif_variant == 'mean':
                importance = self.calculate_actif_mean(aggregated_attributions.cpu().numpy())
            elif actif_variant == 'meanstd':
                importance = self.calculate_actif_meanstddev(aggregated_attributions.cpu().numpy())
            elif actif_variant == 'inv':
                importance = self.calculate_actif_inverted_weighted_mean(aggregated_attributions.cpu().numpy())
            elif actif_variant == 'robust':
                importance = self.calculate_actif_robust(aggregated_attributions.cpu().numpy())
            else:
                raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

        else:
            logging.error("No data processed in the validation loader for ablation.")
            return None

        # Ensure that the 'importance' variable is a list or array with the same length as the number of features (cols)
        if not isinstance(importance, np.ndarray):
            importance = np.array(importance)  # Convert to numpy array if it isn't one already

        if importance.shape[0] != len(self.selected_features):
            raise ValueError(
                f"ACTIF method returned {importance.shape[0]} importance scores, but {len(self.selected_features)} features are expected."
            )

        # Prepare the results with the aggregated importance
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]
        return results

    '''
        Deep Lift
    '''

    def compute_deeplift(self, valid_loader, baseline_type='zero', actif_variant='mean'):
        """
        Computes feature importance using DeepLIFT and aggregates it based on the selected ACTIF variant.
        """
        # List to accumulate attributions
        all_attributions = []
        total_instances = 0
        self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")

        self.currentModel.eval()  # Put the model into evaluation mode

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
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")

            # Initialize DeepLIFT with the model
            explainer = DeepLift(self.currentModel)
            # Compute attributions using DeepLIFT with the given baseline
            attributions = explainer.attribute(inputs, baselines=baselines)

            # Sum across the time steps (dim=1), keeping the batch dimension intact
            attributions_mean = attributions.sum(
                dim=1)  # Sum over the time steps, results in shape [batch_size, num_features]

            # After processing the batch, free GPU memory
            del inputs
            del attributions
            torch.cuda.empty_cache()

            # Append the batch attributions
            all_attributions.append(attributions_mean.detach().cpu().numpy())
            total_instances += attributions_mean.size(0)

        # Concatenate all attributions (to handle batches)
        if total_instances > 0:
            aggregated_attributions = np.concatenate(all_attributions, axis=0)  # Shape: [num_samples, num_features]

            # Now, apply the selected ACTIF variant for feature importance aggregation
            if actif_variant == 'mean':
                importance = self.calculate_actif_mean(aggregated_attributions)
            elif actif_variant == 'meanstd':
                importance = self.calculate_actif_meanstddev(aggregated_attributions)
            elif actif_variant == 'inv':
                importance = self.calculate_actif_inverted_weighted_mean(aggregated_attributions)
            elif actif_variant == 'robust':
                importance = self.calculate_actif_robust(aggregated_attributions)
            else:
                raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

            # Store the attributions as a dataframe for processing
            attributions_df = pd.DataFrame(importance, index=self.selected_features)
            # Compute the mean absolute attributions for each feature
            mean_abs_attributions = attributions_df.abs().mean()
            # Sort the features by their mean absolute attributions
            feature_importance = mean_abs_attributions.sort_values(ascending=False)

            # Return the feature importance as a list of dictionaries
            results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                       feature_importance.items()]
            return results
        else:
            print("No batches processed.")
            return None

    # def compute_deeplift(self,   valid_loader, device, baseline_type='zero'):
    #     accumulated_attributions = torch.zeros(10, len(self.selected_features), device=device)
    #     total_batches = 0
    #
    #     trained_model.eval()  # Put model in evaluation mode
    #
    #     for inputs, _ in valid_loader:
    #         inputs = inputs.to(device)
    #
    #         # Define the baseline based on the selected baseline_type
    #         if baseline_type == 'zero':
    #             baselines = torch.zeros_like(inputs)  # Zero baseline
    #         elif baseline_type == 'random':
    #             baselines = torch.rand_like(inputs)  # Random baseline (uniform between 0 and 1)
    #         elif baseline_type == 'mean':
    #             # Compute the mean baseline (per feature) along dimension 0
    #             mean_baseline = torch.mean(inputs, dim=0)
    #
    #             # Expand the mean_baseline to match the shape of the input
    #             baselines = mean_baseline.expand_as(inputs)
    #             # baselines = torch.full_like(inputs, torch.mean(inputs, dim=0).)  # Mean baseline (per feature)
    #         else:
    #             raise ValueError(f"Unknown baseline type: {baseline_type}")
    #
    #         explainer = DeepLift(trained_model)  # Initialize DeepLIFT with the model
    #         attributions = explainer.attribute(inputs, baselines=baselines)  # Use the chosen baseline
    #         accumulated_attributions += attributions.sum(dim=0)
    #         total_batches += 1
    #
    #     if total_batches > 0:
    #         # Calculate the mean attributions across batches
    #         mean_attributions = (accumulated_attributions / total_batches).detach().cpu().numpy()
    #
    #         # Store the attributions as a dataframe
    #         attributions_df = pd.DataFrame(mean_attributions, columns=self.selected_features)
    #         mean_abs_attributions = attributions_df.abs().mean()
    #         feature_importance = mean_abs_attributions.sort_values(ascending=False)
    #
    #         # Return the feature importance results
    #         results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
    #                    feature_importance.items()]
    #         return results
    #     else:
    #         print("No batches processed.")
    #         return None

    '''
        NISP
    '''

    def compute_nisp(self, valid_loader, version='v1', actif_variant='mean'):
        if version == 'v1':
            return self.compute_nisp_configured(valid_loader, accumulation_steps=1,
                                                use_mixed_precision=False, actif_variant=actif_variant)
        elif version == 'v2':
            return self.compute_nisp_configured(valid_loader, accumulation_steps=1,
                                                use_mixed_precision=True, actif_variant=actif_variant)

        elif version == 'v3':
            return self.compute_nisp_v3(valid_loader, accumulation_steps=1,
                                        use_mixed_precision=False, actif_variant=actif_variant)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_nisp_v3(self, valid_loader, accumulation_steps=1, use_mixed_precision=False,
                        actif_variant='mean'):
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

    def compute_nisp_configured(self, valid_loader, accumulation_steps=1,
                                use_mixed_precision=False, actif_variant='mean'):

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

    def compute_intgrad(self, valid_loader, version='v1', actif_variant='mean'):
        if version == 'v1':
            return self.compute_intgrad_configured(valid_loader, baseline_type='zeroes', actif_variant=actif_variant)
        elif version == 'v2':
            return self.compute_intgrad_configured(valid_loader, baseline_type='random', actif_variant=actif_variant)
        elif version == 'v3':
            return self.compute_intgrad_configured(valid_loader, baseline_type='mean', actif_variant=actif_variant)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_intgrad_configured(self, valid_loader, baseline_type='zeroes', steps=100, actif_variant='mean'):
        # Initialize accumulated attributions with the number of features
        accumulated_attributions = torch.zeros(len(self.selected_features), device=device)
        all_attributions = []  # To store the attributions for each batch
        total_batches = 0

        # Load the model and switch to evaluation mode
        self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")
        self.currentModel.eval()

        # Loop over validation loader
        for inputs, _ in valid_loader:
            inputs = inputs.to(device)

            # Define baselines based on the type
            if baseline_type == 'zeroes':
                baseline = torch.zeros_like(inputs)
            elif baseline_type == 'random':
                baseline = torch.randn_like(inputs)
            elif baseline_type == 'mean':
                baseline = torch.mean(inputs, dim=0, keepdim=True).expand_as(inputs)  # Broadcasting mean baseline
            else:
                raise ValueError(f"Unsupported baseline type: {baseline_type}")

            # Create Integrated Gradients explainer using self.currentModel
            explainer = IntegratedGradients(lambda input_batch: self.currentModel(input_batch))

            # Calculate attributions
            with torch.no_grad():  # Disable gradient computation
                attributions = explainer.attribute(inputs, baselines=baseline, n_steps=steps)
            print("Shape of attributions:", attributions.shape)  # Should still be (batch_size, time_steps, features)

            # Sum over the batch (axis=0) and time (axis=1) dimensions
            summed_attributions = attributions.sum(dim=(0, 1))  # Reduce both batch and time dimensions
            all_attributions.append(summed_attributions.cpu().numpy())  # Convert to NumPy and store
            total_batches += 1

        # Final processing of attributions
        if total_batches > 0:
            # Concatenate all attributions (to handle batches)
            aggregated_attributions = np.concatenate(all_attributions, axis=0)  # Shape: [num_samples, num_features]

            # Apply the selected ACTIF variant for feature importance aggregation
            if actif_variant == 'mean':
                importance = self.calculate_actif_mean(aggregated_attributions)
            elif actif_variant == 'meanstd':
                importance = self.calculate_actif_meanstddev(aggregated_attributions)
            elif actif_variant == 'inv':
                importance = self.calculate_actif_inverted_weighted_mean(aggregated_attributions)
            elif actif_variant == 'robust':
                importance = self.calculate_actif_robust(aggregated_attributions)
            else:
                raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

            # Store the attributions as a dataframe for processing
            attributions_df = pd.DataFrame(importance, index=self.selected_features)
            # Compute the mean absolute attributions for each feature
            mean_abs_attributions = attributions_df.abs().mean()
            # Sort the features by their mean absolute attributions
            feature_importance = mean_abs_attributions.sort_values(ascending=False)

            # Return the feature importance as a list of dictionaries
            results = [{'feature': feature, 'attribution': attribution} for feature, attribution in
                       feature_importance.items()]
            return results
        else:
            print("No batches processed.")
            return None

    '''
        SHAP Values
    '''

    def compute_shap(self, valid_loader, version='v1', actif_variant='mean'):
        # Define the baseline based on the selected baseline_type
        if version == 'v1':
            return self.compute_shap_configured(valid_loader, background_size=10, nsamples=100)
        elif version == 'v2':
            return self.compute_shap_configured(valid_loader, background_size=50, nsamples=300)
        elif version == 'v3':
            return self.compute_shap_configured(valid_loader, background_kmeans=True, nsamples=500)
        else:
            raise ValueError(f"Unknown baseline type: {version}")

    def compute_shap_configured(self, valid_loader, background_size=20, nsamples=100, background_kmeans=False,
                                actif_variant='mean'):

        self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")

        shap_values_accumulated = []

        for input_batch, _ in valid_loader:
            input_batch = input_batch.to(device)

            # Background data sampling
            if background_kmeans:
                background_data = shap.kmeans(input_batch.cpu().numpy(), background_size).data
            else:
                background_data = input_batch[:background_size].cpu().numpy()

            input_instance = input_batch.cpu().numpy().reshape(input_batch.size(0), -1)
            model_wrapper = PyTorchModelWrapper(self.currentModel, (10, 38))

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

    # SHUFFLING:
    # working fine for dynamic model loading, actif variants, no sensitivity possible?
    def feature_shuffling_importances(self, valid_loader, actif_variant='mean'):
        """
        Compute feature importances using feature shuffling and apply ACTIF aggregation.

        Args:
            valid_loader: DataLoader for validation data (with sequences).
            actif_variant: The ACTIF variant to use for aggregation ('mean', 'meanstddev', 'inv',  'robust').

        Returns:
            List of feature importance based on feature shuffling and the selected ACTIF variant.
        """

        self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")

        cols = self.selected_features
        device = next(self.currentModel.parameters()).device  # Ensure we are using the correct device

        # Calculate the baseline MAE (Mean Absolute Error)
        overall_baseline_mae, _ = self.calculateBaseLine(self.currentModel, valid_loader)

        results = [{'feature': 'BASELINE', 'attribution': overall_baseline_mae}]
        self.currentModel.eval()

        # Initialize array to accumulate attributions across all samples and features
        num_samples = len(valid_loader.dataset)  # Total number of samples in validation dataset
        all_attributions = np.zeros((num_samples, len(cols)))  # Shape: (samples_size, features)

        sample_idx = 0  # To track the global sample index across batches

        # Iterate through each feature and compute attributions via shuffling
        for k in tqdm(range(len(cols)), desc="Computing feature importance"):
            # Loop through batches in the validation loader
            for X_batch, y_batch in valid_loader:
                batch_size = X_batch.size(0)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Shuffle the k-th feature for each sample in the batch
                X_batch_shuffled = X_batch.clone()
                indices = torch.randperm(X_batch.size(0))
                X_batch_shuffled[:, :, k] = X_batch_shuffled[indices, :, k]

                # Disable gradient calculation during evaluation
                with torch.no_grad():
                    oof_preds_shuffled = self.currentModel(X_batch_shuffled, return_intermediates=False).squeeze()
                    # Calculate MAE for shuffled predictions
                    attribution_as_mae = torch.mean(torch.abs(oof_preds_shuffled - y_batch), dim=1).cpu().numpy()

                # Ensure we don't exceed the size of all_attributions
                end_idx = sample_idx + batch_size
                if end_idx > num_samples:
                    end_idx = num_samples

                # Store attributions for the current batch and feature k
                all_attributions[sample_idx:end_idx, k] = attribution_as_mae[:end_idx - sample_idx]

                # Update the global sample index
                sample_idx += batch_size

        # Check the shape of all_attributions to ensure it's (samples_size, features)
        print(f"Shape of all_attributions: {all_attributions.shape}")

        # Apply ACTIF variant for aggregation based on the 'actif_variant' parameter
        if actif_variant == 'mean':
            importance = self.calculate_actif_mean(all_attributions)
        elif actif_variant == 'meanstd':
            importance = self.calculate_actif_meanstddev(all_attributions)
        elif actif_variant == 'inv':
            importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        elif actif_variant == 'robust':
            importance = self.calculate_actif_robust(all_attributions)
        else:
            raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

        # Ensure that the importance variable is a list or array with the same length as the number of features
        if not isinstance(importance, np.ndarray):
            importance = np.array(importance)

        if importance.shape[0] != len(cols):
            raise ValueError(
                f"ACTIF method returned {importance.shape[0]} importance scores, but {len(cols)} features are expected."
            )

        # Prepare the results with the aggregated importance
        results = [{'feature': cols[i], 'attribution': importance[i]} for i in range(len(cols))]

        return results

    '''
       =======================================================================================
       # Utility Functions
       =======================================================================================
       '''

    def model_wrapper(self, model, input_tensor):
        output = model(input_tensor, return_intermediates=False)
        return output.squeeze(-1)

    def calculate_memory_and_execution_time(self, method_func):
        start_time = timeit.default_timer()
        mem_usage, subject_importances = memory_usage((method_func,), retval=True, interval=0.1, timeout=None)
        execution_time = timeit.default_timer() - start_time
        logging.info(f"Execution time: {execution_time} seconds")
        logging.info(f"Memory usage: {max(mem_usage)} MiB")
        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {max(mem_usage)} MiB\n")

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
            print(f"Average execution time for {method}: {average_time} seconds")
            print(f"Average memory usage for {method}: {average_memory} MiB \n\n")
            print("=======================================================")

        df_timing = pd.DataFrame(self.timing_data)
        file_path = f"{self.outputFolder}/method_execution_times.csv"
        header = not os.path.exists(file_path)
        df_timing.to_csv(file_path, mode='a', index=False, header=header)
        logging.info(f"Appended average execution times to '{file_path}'")

    def sort_importances_based_on_attribution(self, aggregated_importances, method):
        # Create DataFrame from aggregated importances
        # if isinstance(aggregated_importances, list):
        df_importances = pd.DataFrame(aggregated_importances)

        # Check if DataFrame is empty or doesn't have required columns
        if df_importances.empty:
            raise ValueError(f"No feature importances found for method {method}")
        if 'feature' not in df_importances.columns or 'attribution' not in df_importances.columns:
            raise KeyError("The DataFrame does not contain the required 'feature' and 'attribution' columns.")

        # Group by 'feature' and compute the mean of the attributions
        mean_importances = df_importances.groupby('feature')['attribution'].mean().reset_index()

        # Sort the importances by attribution values
        sorted_importances = mean_importances.sort_values(by='attribution', ascending=False)
        self.save_importances_in_file(mean_importances_sorted=sorted_importances, method=method)
        # Store the sorted importances
        self.feature_importance_results[method] = sorted_importances
        return sorted_importances

    def save_importances_in_file(self, mean_importances_sorted, method):
        filename = f"{self.outputFolder}/{method}.csv"
        mean_importances_sorted.to_csv(filename, index=False)
        logging.info(f"Saved importances for {method} in {filename}")

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

    def loadFOVALModel(self, model_path, featureCount=38):
        jsonFile = model_path + '.json'

        with open(jsonFile, 'r') as f:
            hyperparameters = json.load(f)

        model = FOVAL(input_size=featureCount, embed_dim=hyperparameters['embed_dim'],
                      fc1_dim=hyperparameters['fc1_dim'],
                      dropout_rate=hyperparameters['dropout_rate']).to(device)

        return model, hyperparameters
