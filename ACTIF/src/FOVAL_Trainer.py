import pickle
import random
from collections import defaultdict
import csv
import os
from matplotlib.gridspec import GridSpec
import keyboard  # Import the keyboard library
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

import wandb
import torch
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim

from FOVAL_Preprocessor import subjective_normalization_dataset, separate_features_and_targets
from RobustVision_Dataset import create_sequences
from SimpleLSTM import SimpleLSTM_V2
from Utilities import create_lstm_tensors_dataset, create_dataloaders_dataset, define_model_and_optim, analyzeResiduals, \
    visualizeSingleSubjectPredictions, print_results

pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU
n_epochs = 2000
# n_epochs = 2
# n_epochs = 100
# n_epochs = 500


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


class FOVAL_Trainer:
    pd.option_context('mode.use_inf_as_na', True)

    def __init__(self):


        self.SHOW_RESIDUAL_DIST = True
        self.input_size = None
        self.train_loader = None
        self.valid_loader = None
        self.rv_dataset = None
        # Define the CSV filename
        self.all_predictions_array = None
        self.all_true_values_array = None
        self.userFolder = ""
        self.fold_results = None
        self.best_test_smae = float('inf')
        self.best_test_mse = float('inf')
        self.best_test_mae = float('inf')
        self.test_features_input = None
        self.test_targets_input = None

        self.dataset_augmented = None
        self.aggregated_data = None
        self.dataset = None
        self.GOAL_SMAE = 40
        self.training_features_input = None
        self.training_targets_input = None
        self.validation_features_input = None
        self.validation_targets_input = None
        self.scheduler = None
        self.target_transformer = None
        self.optimizer = None
        self.patience_limit = 1000
        self.early_stopping_threshold = 1.0  # Set the threshold for early stopping
        self.csv_filename = "training_results.csv"
        self.val_dataset = None
        self.iteration_counter = 0
        self.train_dataset = None
        self.path_to_save_model = "."

        self.history_smae_val = []
        self.history_smae_train = []
        self.history_mae_val = []
        self.history_mse_val = []
        self.history_mse_train = []
        self.history_mae_train = []
        self.history_mae = []
        self.history_mae_train = []
        self.history_mse_train = []

        self.avg_val_mse = float('inf')
        self.avg_val_mae = float('inf')
        self.avg_val_rmse = float('inf')
        self.avg_val_smae = float('inf')
        self.avg_val_r2 = -1000
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_smae = float('inf')
        self.avg_train_mse = float('inf')
        self.avg_train_mae = float('inf')
        self.avg_train_rmse = float('inf')
        self.avg_train_smae = float('inf')
        self.best_train_mse = float('inf')
        self.best_train_mae = float('inf')
        self.best_train_smae = float('inf')

        self.target_scaler = None

        self.average_importances = None
        self.all_importances = None
        self.running_number = None
        self.validation_targets = None
        self.training_targets = None
        self.validation_data = None
        self.training_data = None
        self.giw_data = None
        self.average_estimated_depths = None
        self.name = "DepthEstimator"
        self.input_data = None
        self.model = None
        print("Device is: ", device)
        self.target_transformation = None
        self.transformers = None
        self.sequence_length = None
        self.global_metrics = defaultdict(lambda: defaultdict(list))

        self.global_error_distribution = defaultdict(list)
        self.global_depth_bins_mae = defaultdict(list)
        self.global_specific_range_error = []

    def convert_to_tensors(self):
        # Convert target arrays to tensors and send to device
        y_train_array = np.array(self.training_targets)
        y_train_nn = torch.tensor(y_train_array, dtype=torch.float32).reshape(-1, 1).to(device)

        y_validation_array = np.array(self.validation_targets)
        y_test_nn = torch.tensor(y_validation_array, dtype=torch.float32).reshape(-1, 1).to(device)

        # Convert feature arrays to tensors and send to device
        X_train_nn = torch.tensor(self.training_data, dtype=torch.float32).to(device)
        X_test_nn = torch.tensor(self.validation_data, dtype=torch.float32).to(device)

        return X_train_nn, y_train_nn, X_test_nn, y_test_nn

    def calculateBaseLine(self, model, val_loader):
        mae_loss_fn = nn.L1Loss()
        baseline_maes = []

        intermediate_activations=None
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, intermediate_activations = model(X_batch, return_intermediates=True)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                # calculate loss
                val_mae = mae_loss_fn(y_pred, y_batch).item()

                print("Val MAE: ", val_mae)
                baseline_maes.append(val_mae)

        overall_baseline_mae = np.mean(baseline_maes)
        return overall_baseline_mae, intermediate_activations, y_batch

    def calculate_averages(self):
        averaged_results = {
            'error_distribution': {},
            'mean_errors_per_bin': {},
            'average_error_for_2m_range': 0,
        }

        with open('error_distribution_percentage.pkl', 'wb') as file:
            pickle.dump(self.global_error_distribution, file)

        # Calculate averages for error distributions
        for key, values in self.global_error_distribution.items():
            averaged_results['error_distribution'][key] = sum(values) / len(values)

        with open('depth_bins_mae.pkl', 'wb') as file:
            pickle.dump(self.global_depth_bins_mae, file)

        # Calculate averages for mean errors per bin
        for bin_key, errors in self.global_depth_bins_mae.items():
            averaged_results['mean_errors_per_bin'][bin_key] = sum(errors) / len(errors)

        with open('two_meter_range_MAE.pkl', 'wb') as file:
            pickle.dump(self.global_specific_range_error, file)

        # Calculate average for the specific range error
        averaged_results['average_error_for_2m_range'] = sum(self.global_specific_range_error) / len(
            self.global_specific_range_error)

        return averaged_results

    def update_global_metrics(self, new_results):
        print("updating...")
        # Update error distribution percentages
        for key in ['<1cm', '1-10cm', '10-20', '>20']:
            self.global_error_distribution[key].append(new_results[key])

        # Update mean errors per bin
        for bin_key, bin_error in new_results['mean_errors_per_bin'].items():
            self.global_depth_bins_mae[bin_key].append(bin_error)

        # Update the average error for a specific range (if applicable)
        # Assuming you have a global structure for this as well
        self.global_specific_range_error.append(new_results['average_error_for_2m_range'])

    def train_epoch(self, model, optimizer, train_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn, epoch):
        # Training code for one epoch here
        model.train()
        total_samples = 0.0
        all_y_true = []
        all_y_pred = []
        total_mae = 0
        total_mse = 0
        total_smae = 0

        for i, (X_batch, y_batch) in enumerate(train_loader):
            # Listen for the "q" key press event
            if keyboard.is_pressed('q'):
                print("Pressed q, skipping epochs.")
                break  # Exit the inner loop to stop training

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred, intermediate_activations = model(X_batch, return_intermediates=True)
            if torch.isnan(y_pred).any():
                raise ValueError('NaN values in model output')

            # Calculate loss on the scaled data
            smae_loss = smae_loss_fn(y_pred, y_batch)

            combined_loss = smae_loss
            combined_loss.backward()
            optimizer.step()

            # Inverse transform for metric calculation (post-backpropagation)
            y_pred_inv = self.inverse_transform_target(y_pred).to(device)
            y_batch_inv = self.inverse_transform_target(y_batch).to(device)

            # Accumulate metrics on the original scale
            total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_mse += mse_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_smae += smae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_samples += y_batch.size(0)

            all_y_true.append(y_batch_inv.detach().cpu().numpy())
            all_y_pred.append(y_pred_inv.detach().cpu().numpy())

        avg_train_mae = total_mae / total_samples
        avg_train_mse = total_mse / total_samples
        avg_train_smae = total_smae / total_samples
        avg_train_rmse = np.sqrt(avg_train_mse)
        avg_train_r2 = 1

        # wandb.log({
        #     'avg_train_mae': avg_train_mae,
        #     'avg_train_mse': avg_train_mse,
        #     'avg_train_smae': avg_train_smae,
        #     'avg_train_rmse': avg_train_rmse,
        # })

        # for every epoch save activations
        # self.saveActivations(self.userFolder, intermediate_activations, epoch=epoch, batch_idx=i)

        self.checkTrainResults(avg_train_mse, avg_train_mae, avg_train_smae, avg_train_rmse, avg_train_r2)
        return avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2

    def save_activations_validation(self, intermediates, target_vector, name, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        activations_data = {}  # Dictionary to store activations data

        # Save intermediate activations
        for i, (key, activation) in enumerate(intermediates.items()):
            # Convert the tensor to numpy array and save it
            activations_data[key] = activation.detach().cpu().numpy()

        # Save target vector
        activations_data['target_vector'] = target_vector.detach().cpu().numpy()

        # Save the activations data dictionary
        activations_file = os.path.join(save_dir, f'{name}_activations.npy')
        np.save(activations_file, activations_data)

        return activations_file

    def visualize_activations(self, intermediates, target_vector, userfolder, name):
        self.save_activations_validation(intermediates, target_vector, name, userfolder)

        fig = plt.figure(figsize=(10, 12), layout="constrained")
        gs = GridSpec(6, 2, figure=fig)

        # Plot intermediate activations
        for i, (key, activation) in enumerate(intermediates.items()):
            if activation.ndim > 2:
                activation = activation.mean(dim=1)  # Taking the mean across time steps if applicable
                # activation = np.mean(activation, axis=1)  # Averaging if more than 2 dimensions

            # Plot only a subset of samples for clarity
            activation_subset = activation[:10, :]

            if i == 0:
                ax0 = fig.add_subplot(gs[0, :])

                # Plot intermediate activations spanning both columns
                im = ax0.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax0.set_title(f'{key}', pad=10, fontsize=12)
                ax0.set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=ax0)
                ax0.tick_params(axis='both', which='major', labelsize=8)

                # Set the x-axis labels and ticks to the bottom
                ax0.xaxis.set_ticks_position('bottom')
                ax0.xaxis.set_label_position('bottom')

            if i == 1:
                ax1 = fig.add_subplot(gs[1, :])

                # Plot intermediate activations spanning both columns
                im = ax1.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax1.set_title(f'{key}', pad=10, fontsize=12)
                ax1.set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=ax1)
                ax1.tick_params(axis='both', which='major', labelsize=8)
                # Set the x-axis labels and ticks to the bottom
                ax1.xaxis.set_ticks_position('bottom')
                ax1.xaxis.set_label_position('bottom')

            if i == 2:
                ax2 = fig.add_subplot(gs[2, :])

                # Plot intermediate activations spanning both columns
                im = ax2.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax2.set_title(f'{key}', pad=10, fontsize=12)
                ax2.set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=ax2)
                ax2.tick_params(axis='both', which='major', labelsize=8)
                # Set the x-axis labels and ticks to the bottom
                ax2.xaxis.set_ticks_position('bottom')
                ax2.xaxis.set_label_position('bottom')

            if i == 3:
                ax3 = fig.add_subplot(gs[3, :])

                # Plot intermediate activations spanning both columns
                im = ax3.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax3.set_title(f'{key}', pad=10, fontsize=12)
                ax3.set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=ax3)
                ax3.tick_params(axis='both', which='major', labelsize=8)
                # Set the x-axis labels and ticks to the bottom
                ax3.xaxis.set_ticks_position('bottom')
                ax3.xaxis.set_label_position('bottom')

            if i == 4:
                ax4 = fig.add_subplot(gs[4, :])

                # Plot intermediate activations spanning both columns
                im = ax4.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax4.set_title(f'{key}', pad=10, fontsize=12)
                ax4.set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=ax4)
                ax4.tick_params(axis='both', which='major', labelsize=8)
                # Set the x-axis labels and ticks to the bottom
                ax4.xaxis.set_ticks_position('bottom')
                ax4.xaxis.set_label_position('bottom')

            if i == 5:
                ax5 = fig.add_subplot(gs[5, :-1])

                im = ax5.matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                ax5.set_title(f'{key}', pad=10, fontsize=12)
                ax5.set_ylabel('Samples', fontsize=10)
                # fig.colorbar(im, ax=ax5)
                ax5.tick_params(axis='both', which='major', labelsize=8)

                # Hide x-ticks for the last activation plot
                ax5.set_xticks([])

                # target vector

                ax6 = fig.add_subplot(gs[5, -1])
                ax6.set_xticks([])

                # Plot the target vector in the same row as the last intermediate activation
                target_vector_flipped = target_vector[:10].view(-1, 1).detach().cpu().numpy()
                # target_vector_flipped = activations_data['target_vector']
                im_target = ax6.matshow(target_vector_flipped[:10].reshape(-1, 1), aspect='auto', cmap='viridis')

                # im_target = ax5.matshow(target_vector_flipped, aspect='auto', cmap='viridis')
                ax6.set_title('Target Vector', pad=10, fontsize=12)
                # ax6.tick_params(axis='both', which='major', labelsize=8)
                fig.colorbar(im_target, ax=ax6)
                ax6.set_xticks([])

                fig.suptitle("Activation Visualization")
                format_axes(fig)
        # plt.show()

    def visualize_activations_ok(self, intermediates, target_vector):
        fig_width = 8  # Width of the figure
        fig_height_per_layer = 3  # Height per layer

        # Adjust figure height based on the number of layers
        total_fig_height = len(intermediates) * fig_height_per_layer

        # Create a figure with subplots
        fig, axs = plt.subplots(len(intermediates), 2, figsize=(fig_width, total_fig_height),
                                gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)

        # Plot intermediate activations
        for i, (key, activation) in enumerate(intermediates.items()):
            if activation.ndim > 2:
                activation = activation.mean(dim=1)  # Taking the mean across timesteps if applicable

            # Plot only a subset of samples for clarity
            activation_subset = activation[:10, :]

            if i < 4:  # First four rows
                # Plot intermediate activations spanning both columns
                im = axs[i, 0].matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                axs[i, 0].set_title(f'{key}', pad=10, fontsize=12)
                axs[i, 0].set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=axs[i, 0])
                axs[i, 0].tick_params(axis='both', which='major', labelsize=8)

                # Hide x-ticks for all but the last activation plot
                if i > 3:
                    axs[i, 0].set_xticks([])

                # Set the x-axis labels and ticks to the bottom
                axs[i, 0].xaxis.set_ticks_position('bottom')
                axs[i, 0].xaxis.set_label_position('bottom')

                # Remove the empty subfigure on the right
                axs[i, 1].remove()

            else:  # Last row
                im = axs[i, 0].matshow(activation_subset.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                axs[i, 0].set_title(f'{key}', pad=10, fontsize=12)
                axs[i, 0].set_ylabel('Samples', fontsize=10)
                fig.colorbar(im, ax=axs[i, 0])
                axs[i, 0].tick_params(axis='both', which='major', labelsize=8)

                # Hide x-ticks for the last activation plot
                axs[i, 0].set_xticks([])

                # Set the x-axis labels and ticks to the bottom
                axs[i, 0].xaxis.set_ticks_position('bottom')
                axs[i, 0].xaxis.set_label_position('bottom')

        # Plot the target vector in the same row as the last intermediate activation
        target_vector_flipped = target_vector[:10].view(-1, 1).detach().cpu().numpy()
        im_target = axs[-1, 1].matshow(target_vector_flipped, aspect='auto', cmap='viridis')
        axs[-1, 1].set_title('Target Vector', pad=10, fontsize=12)
        axs[-1, 1].tick_params(axis='both', which='major', labelsize=8)
        fig.colorbar(im_target, ax=axs[-1, 1])

        # Set the x-axis labels and ticks to the bottom for the target vector subplot
        axs[-1, 1].xaxis.set_ticks_position('bottom')
        axs[-1, 1].xaxis.set_label_position('bottom')

        # Set the x-axis label for the intermediate activations subplot
        axs[-1, 0].set_xlabel("Activations (learned features)", fontsize=10)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between plots
        plt.savefig(f"{self.userFolder}/Activations_with_Target.png", format='png', dpi=300, bbox_inches='tight')
        # plt.show()

        # # Adjustments for overall figure dimensions
        # fig_width = 12
        # fig_height_per_layer = 2
        # total_fig_height = len(
        #     intermediates) * fig_height_per_layer + fig_height_per_layer  # Adding extra space for output and target vectors
        #
        # # Creating a subplot for each layer's activations plus one for the output vs. target visualization
        # fig, axs = plt.subplots(len(intermediates) + 1, 2, figsize=(fig_width, total_fig_height), squeeze=False,
        #                         gridspec_kw={'width_ratios': [3, 1]})
        #
        # # Visualize all layer activations
        # for i, (key, activation) in enumerate(intermediates.items()):
        #     if activation.ndim > 2:
        #         activation = activation.mean(dim=1)
        #
        #     # Selecting a subset for visualization
        #     activation = activation[:10]
        #
        #     im = axs[i, 0].matshow(activation.detach().cpu().numpy(), aspect='auto', cmap='viridis')
        #     axs[i, 0].set_title(f'{key}')
        #
        #     sample_range = range(1, activation.shape[0] + 1)
        #     axs[i, 0].set_yticks(range(activation.shape[0]))
        #     axs[i, 0].set_yticklabels(sample_range)
        #
        #     axs[i, 0].set_ylabel('Samples')
        #     fig.colorbar(im, ax=axs[i, 0])
        #
        #     # Removing x-ticks and labels for intermediary activations
        #     axs[i, 0].set_xticks([])
        #
        #     # Hide the second column for intermediates
        #     axs[i, 1].axis('off')
        #
        #
        # plt.show()

    def validate_epoch(self, model, val_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn, patience_counter, epoch,
                       all_predictions, all_true_values):

        model.eval()
        all_predictions_array = []
        all_true_values_array = []
        total_val_mae = 0
        total_val_mse = 0
        total_val_smae = 0
        total_val_samples = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, intermediates = model(X_batch, return_intermediates=True)

                # y_pred = model(X_batch)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                # calculate loss
                val_mse = mse_loss_fn(y_pred, y_batch)
                val_mae = mae_loss_fn(y_pred, y_batch)
                val_smae = smae_loss_fn(y_pred, y_batch)

                # accumulate loss
                total_val_mae += val_mae.item() * y_batch.size(0)
                total_val_mse += val_mse.item() * y_batch.size(0)
                total_val_smae += val_smae.item() * y_batch.size(0)

                # count samples of batch
                total_val_samples += y_batch.size(0)

                # Assume errors is a list of absolute errors for each prediction
                errors = (y_pred - y_batch).cpu().numpy()

                # Log the biggest and smallest error
                max_error = np.max(errors)
                min_error = np.min(errors)
                median_error = np.median(errors)

                # Store predictions and true values
                all_predictions.append(y_pred.detach().cpu().numpy())
                all_true_values.append(y_batch.detach().cpu().numpy())

                # Logging with wandb
                # wandb.log({
                #     'max_error_val_batch': max_error,
                #     'min_error_val_batch': min_error,
                #     'median_error_val_batches': median_error
                # })

        try:
            all_predictions_array = np.concatenate(all_predictions)
            all_true_values_array = np.concatenate(all_true_values)
        except ValueError as e:
            # Log the error and the state of relevant variables
            print(f"Error: {e}")
            print(f"all_predictions length: {len(all_predictions)}")
            print(f"all_true_values length: {len(all_true_values)}")
            if all_predictions:
                print(f"Shape of first element in all_predictions: {all_predictions[0].shape}")
            if all_true_values:
                print(f"Shape of first element in all_true_values: {all_true_values[0].shape}")

        # Now calculate residuals
        residuals = np.abs(all_predictions_array - all_true_values_array)
        raw_residuals = all_predictions_array - all_true_values_array
        # wandb.log({'val_batch_residuals': residuals})

        # Calculate R-squared for the entire dataset
        avg_val_r2 = r2_score(all_true_values_array, all_predictions_array)

        # EPOCH LOSSES
        avg_val_mae = total_val_mae / total_val_samples
        avg_val_mse = total_val_mse / total_val_samples
        avg_val_smae = total_val_smae / total_val_samples
        avg_val_rmse = np.sqrt(avg_val_mse)

        # wandb.log({'avg_val_mse': avg_val_mse,
        #            'avg_val_mae': avg_val_mae,
        #            'avg_val_smae': avg_val_smae,
        #            'avg_val_rmse': avg_val_rmse,
        #            'avg_val_r2': avg_val_r2
        #            })

        isBreakLoop, patience_counter = self.checkValidationResults(avg_val_mse, avg_val_mae, avg_val_smae, epoch,
                                                                    patience_counter, all_predictions_array,
                                                                    all_true_values_array)

        if (epoch >= n_epochs - 1) or isBreakLoop:
            # Visualize the activations
            self.visualize_activations(intermediates, y_batch, self.userFolder, "last_validation")
            # self.update_global_metrics(self.fold_results)
            analyzeResiduals(self.all_predictions_array, self.all_true_values_array)
            # visualizeSingleSubjectPredictions(self.userFolder, self.all_true_values_array, self.all_predictions_array)

            self.fold_results = None

        if isBreakLoop:
            print(f"Early stopping reached limit of {self.patience_limit}")
            return True, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals

        return False, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals

    def runFold(self, batch_size, embed_dim, learning_rate, weight_decay, l1_lambda, dropoutRate, fc1_dim, fold_count,
                n_splits, train_index, val_index, test_index, fold_performance, model=None, beta=0.5, method=None):

        # Choose original or augmented training data for this fold
        print(f"Fold {fold_count}/{n_splits}")

        train_loader, valid_loader, test_loader, input_size = self.get_data_loader(train_index, val_index, test_index,
                                                                                   batch_size=batch_size)

        # Set target scaler for this object
        self.target_scaler = self.rv_dataset.target_scaler
        print("Target scaler is: ", self.target_scaler)
        print("Created data loaders")

        self.model, self.optimizer, self.scheduler = define_model_and_optim(model=model, input_size=input_size,
                                                                            embed_dim=embed_dim,
                                                                            dropoutRate=dropoutRate,
                                                                            learning_rate=learning_rate,
                                                                            weight_decay=weight_decay,
                                                                            fc1_dim=fc1_dim)
        print(self.model)
        print("Created model and optim")
        isContinueFold, goToNextOptimStep, self.best_val_mae = self.runConfiguredFold(batch_size,
                                                                                      embed_dim,
                                                                                      dropoutRate,
                                                                                      l1_lambda,
                                                                                      learning_rate,
                                                                                      weight_decay,
                                                                                      fc1_dim,
                                                                                      fold_performance,
                                                                                      train_loader,
                                                                                      valid_loader,
                                                                                      test_loader,
                                                                                      beta=beta,
                                                                                      method=method)

        return isContinueFold, goToNextOptimStep, self.best_val_mae

    def checkTrainResults(self, avg_train_mse, avg_train_mae, avg_train_smae, avg_train_rmse, avg_train_r2):

        # Check training results
        if float(avg_train_mae) < self.best_train_mae:
            self.best_train_mae = avg_train_mae

        if float(avg_train_smae) < self.best_train_smae:
            self.best_train_smae = avg_train_smae

        """ 3. Implement early stopping """
        if float(avg_train_mse) < self.best_train_mse:
            self.best_train_mse = avg_train_mse

    def checkValidationResults(self, avg_val_mse, avg_val_mae, avg_val_smae, epoch, patience_counter,
                               all_predictions_array, all_true_values_array):
        isBreakLoop = False
        # Check validation results
        if avg_val_mae < self.best_val_mae:
            self.best_val_mae = avg_val_mae

        if avg_val_smae < self.best_val_smae:
            self.best_val_smae = avg_val_smae
            # torch.save(self.model.state_dict(), 'best_model_state_dict.pth')
            torch.save(self.model.state_dict(), 'results/best_model_state_dict.pth')

            patience_counter = 0
            self.fold_results = analyzeResiduals(all_predictions_array, all_true_values_array)
            self.all_predictions_array = all_predictions_array
            self.all_true_values_array = all_true_values_array
            print(
                f'Model saved at epoch {epoch} with validation SMAE {self.best_val_smae:.6f} and MAE {self.best_val_mae}\n')
        else:
            patience_counter += 1

        """ 3. Implement early stopping """
        if avg_val_mse < self.best_val_mse:
            self.best_val_mse = avg_val_mse

        if avg_val_smae < self.early_stopping_threshold:
            isBreakLoop = True

        if patience_counter > self.patience_limit:
            isBreakLoop = True

        # wandb.log({'best_val_mae': self.best_val_mae, 'best_val_smae': self.best_val_smae,
        #            'best_val_mse': self.best_val_mse})

        return isBreakLoop, patience_counter

    def inverse_transform_target(self, y_transformed):
        # Move the tensor to CPU if it's on GPU
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        # Now that the tensor is on the CPU, convert it to a NumPy array
        y_transformed_np = y_transformed.detach().numpy()

        # Reshape the array to a 2D array with a single column
        y_transformed_np_reshaped = y_transformed_np.reshape(-1, 1)

        if self.target_scaler is not None:
            # Perform the inverse transformation using the scaler
            y_inverse_transformed = self.target_scaler.inverse_transform(y_transformed_np_reshaped)

            # Flatten the array back to 1D
            y_inverse_transformed = y_inverse_transformed.flatten()

            # Convert the NumPy array back to a tensor
            return torch.from_numpy(y_inverse_transformed).to(device)
        else:
            return y_transformed

    def runConfiguredFold(self, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay, fc1_dim,
                          fold_performance, train_loader_0, valid_loader_0=None, test_loader_0=None, beta=0.5,
                          method=None):
        goToNextOptimStep = False

        mse_loss_fn = nn.MSELoss(reduction='sum').to(device)
        mae_loss_fn = nn.L1Loss().to(device)
        smae_loss_fn = nn.SmoothL1Loss(beta=beta).to(device)

        isBreakLoop = False
        avg_train_r2 = -1000
        patience_counter = 0
        avg_train_mse = None
        avg_train_rmse = None
        avg_train_smae = None
        avg_train_mae = None

        avg_val_mse = None
        avg_val_rmse = None
        avg_val_mae = None
        avg_val_smae = None
        avg_val_r2 = None

        self.best_train_mse = float('inf')
        self.best_train_mae = float('inf')
        self.best_train_smae = float('inf')
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_smae = float('inf')

        for epoch in range(n_epochs):
            # # wandb.log({'epochs': epoch})
            # Error analysis
            all_predictions = []
            all_true_values = []

            # train one epoch
            avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2 = self.train_epoch(
                self.model,
                self.optimizer,
                train_loader_0,
                mse_loss_fn,
                mae_loss_fn,
                smae_loss_fn,
                epoch)

            if keyboard.is_pressed('q'):
                goToNextOptimStep = True
                isBreakLoop = True
                break  # Exit the outer loop to stop training completely for current subject

            if valid_loader_0 is not None:
                isBreakLoop, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals = self.validate_epoch(
                    model=self.model, val_loader=valid_loader_0, mse_loss_fn=mse_loss_fn, mae_loss_fn=mae_loss_fn,
                    smae_loss_fn=smae_loss_fn, patience_counter=patience_counter, epoch=epoch,
                    all_predictions=all_predictions, all_true_values=all_true_values)

            if test_loader_0 is not None:
                print("Analysis of residuals for test data")
                all_true_values_array, all_predictions_array = self.test_new_data(model=self.model,
                                                                                  test_loader=test_loader_0,
                                                                                  mse_loss_fn=mse_loss_fn,
                                                                                  mae_loss_fn=mae_loss_fn,
                                                                                  smae_loss_fn=smae_loss_fn)

                if self.SHOW_RESIDUAL_DIST:
                    analyzeResiduals(all_predictions_array, all_true_values_array)
                print("\n\n")

            # Step through the scheduler at the end of each epoch
            self.scheduler.step()

            if isBreakLoop or epoch == n_epochs - 1:
                self.model.load_state_dict(torch.load('results/best_model_state_dict.pth'))
                model_path = os.path.join(self.userFolder, 'optimal_subject_model_state_dict.pth')
                torch.save(self.model.state_dict(), model_path)  # Saving state dictionary
                print("Optimal model state dictionary saved.")
                break

        # Results after all epochs
        print_results(self.iteration_counter, batch_size, embed_dim, dropoutRate, l1_lambda,
                      learning_rate,
                      weight_decay, fc1_dim, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae,
                      avg_train_r2,
                      self.best_train_mse, self.best_train_mae, self.best_train_smae, avg_val_mse,
                      avg_val_rmse,
                      avg_val_mae, avg_val_smae, avg_val_r2, self.best_val_mse, self.best_val_mae,
                      self.best_val_smae)

        # Store the metrics for this fold
        fold_performance.append({
            'fold': len(fold_performance) + 1,
            'avg_train_mse': avg_train_mse,
            'avg_train_rmse': avg_train_rmse,
            'avg_train_mae': avg_train_mae,
            'avg_train_smae': avg_train_smae,
            'avg_train_r2': avg_train_r2,
            'best_train_mse': self.best_train_mse,
            'best_train_mae': self.best_train_mae,
            'best_train_smae': self.best_train_smae,
            'avg_val_mse': avg_val_mse,
            'avg_val_rmse': avg_val_rmse,
            'avg_val_mae': avg_val_mae,
            'avg_val_smae': avg_val_smae,
            'avg_val_r2': avg_val_r2,
            'best_val_mse': self.best_val_mse,
            'best_val_mae': self.best_val_mae,
            'best_val_smae': self.best_val_smae
        })

        # average_fold_val_mae = np.mean([f['best_val_mae'] for f in fold_performance])
        # print(f"Average Validation SMAE across folds: {average_fold_val_mae}")
        # # wandb.log({'average_fold_val_mae': average_fold_val_smae})

        average_fold_val_mae = np.mean([f['best_val_mae'] for f in fold_performance])
        print(f"Average Validation MAE across folds: {average_fold_val_mae}\n")
        # wandb.log({'average_fold_val_mae': average_fold_val_mae})

        if test_loader_0 is not None:
            all_true_values_array, all_predictions_array = self.test_new_data(model=self.model,
                                                                              test_loader=test_loader_0,
                                                                              mse_loss_fn=mse_loss_fn,
                                                                              mae_loss_fn=mae_loss_fn,
                                                                              smae_loss_fn=smae_loss_fn)

            with open('gt_values.pkl', 'wb') as file:
                pickle.dump(all_true_values_array, file)

            with open('pred_values.pkl', 'wb') as file:
                pickle.dump(all_predictions_array, file)

            if self.SHOW_RESIDUAL_DIST:
                analyzeResiduals(all_predictions_array, all_true_values_array)
            # visualizePredictions(all_true_values_array, all_predictions_array)

        return True, goToNextOptimStep, self.best_val_mae

    def test_new_data(self, model, test_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn):

        model.eval()
        all_predictions = []
        all_true_values = []

        total_test_mae = 0
        total_test_mse = 0
        total_test_smae = 0
        total_test_samples = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                # calculate loss
                test_mse = mse_loss_fn(y_pred, y_batch)
                test_mae = mae_loss_fn(y_pred, y_batch)
                test_smae = smae_loss_fn(y_pred, y_batch)

                # accumulate loss
                total_test_mae += test_mae.item() * y_batch.size(0)
                total_test_mse += test_mse.item() * y_batch.size(0)
                total_test_smae += test_smae.item() * y_batch.size(0)

                # count samples of batch
                total_test_samples += y_batch.size(0)

                # Assume errors is a list of absolute errors for each prediction
                errors = (y_pred - y_batch).cpu().numpy()

                # Log the biggest and smallest error
                max_error = np.max(errors)
                min_error = np.min(errors)
                median_error = np.median(errors)

                mode_result = stats.mode(errors, axis=None)
                mode_value = mode_result.mode[0]  # First element of the mode array
                mode_count = mode_result.count[0]  # First element of the count array

                all_predictions.append(y_pred.detach().cpu().numpy())
                all_true_values.append(y_batch.detach().cpu().numpy())

                # Logging with wandb
                # wandb.log({
                #     'max_error_test_batch': max_error,
                #     'min_error_test_batch': min_error
                # })

        try:
            all_predictions_array = np.concatenate(all_predictions)
            all_true_values_array = np.concatenate(all_true_values)
        except ValueError as e:
            # Log the error and the state of relevant variables
            print(f"Error: {e}")
            print(f"all_predictions length: {len(all_predictions)}")
            print(f"all_true_values length: {len(all_true_values)}")
            if all_predictions:
                print(f"Shape of first element in all_predictions: {all_predictions[0].shape}")
            if all_true_values:
                print(f"Shape of first element in all_true_values: {all_true_values[0].shape}")

            # Now calculate residuals
        residuals = np.abs(all_predictions_array - all_true_values_array)

        # wandb.log({'test_batch_residuals': residuals})

        # Calculate R-squared for the entire dataset
        avg_test_r2 = r2_score(all_true_values_array, all_predictions_array)

        # EPOCH LOSSES
        avg_test_mae = total_test_mae / total_test_samples
        avg_test_mse = total_test_mse / total_test_samples
        avg_test_smae = total_test_smae / total_test_samples
        avg_test_rmse = np.sqrt(avg_test_mse)

        print(f"Average test MAE is {avg_test_mae}")

        # wandb.log({'avg_test_mse': avg_test_mse,
        #            'avg_test_mae': avg_test_mae,
        #            'avg_test_smae': avg_test_smae,
        #            'avg_test_rmse': avg_test_rmse,
        #            'avg_test_r2': avg_test_r2
        #            })

        # Check validation results
        if avg_test_mae < self.best_test_mae:
            self.best_test_mae = avg_test_mae

        if avg_test_mse < self.best_test_mse:
            self.best_test_mse = avg_test_mse

        if avg_test_smae < self.best_test_smae:
            self.best_test_smae = avg_test_smae
            if self.SHOW_RESIDUAL_DIST:
                analyzeResiduals(all_predictions_array, all_true_values_array)

        if avg_test_mse < self.best_test_mse:
            self.best_val_mse = avg_test_mse

        # wandb.log({'best_test_mae': self.best_test_mae, 'best_test_smae': self.best_test_smae,
        #           'best_test_mse': self.best_test_mse})

        return all_true_values_array, all_predictions_array

    def get_data_loader(self, train_index, val_index, test_index, batch_size=100):

        valid_loader_0 = None
        test_loader_0 = None

        train_subjects = train_index

        print("Train subjects: ", train_subjects)
        train_data = self.dataset[self.dataset['SubjectID'].isin(train_subjects)]

        train_data = self.rv_dataset.create_features(train_data)

        train_data = subjective_normalization_dataset(train_data)
        train_data = self.rv_dataset.apply_transformation_dataset(train_data, isTrain=True)
        train_data = self.rv_dataset.scale_target_dataset(train_data, isTrain=True)
        assert len(train_data) > 0, "Training data is empty."
        train_sequences = create_sequences(train_data)
        assert len(train_sequences) > 0, "Training sequences are empty."
        train_features, train_targets = separate_features_and_targets(train_sequences)
        assert len(train_features) > 0 and len(train_targets) > 0, "Training features or targets are empty."

        # Example usage
        train_features_tensor, train_targets_tensor = create_lstm_tensors_dataset(train_features, train_targets)

        # get data loaders
        train_loader_0, input_size_0 = create_dataloaders_dataset(train_features_tensor, train_targets_tensor,
                                                                  batch_size=batch_size)
        print("Train features are: ")

        self.train_loader = train_loader_0
        self.input_size = input_size_0

        if val_index is not None:
            # val_subjects = subject_list[val_index]
            val_subjects = val_index
            print("Valid subjects: ", val_subjects)
            val_subjects = [val_subjects] if isinstance(val_subjects, str) else val_subjects
            validation_data = self.dataset[self.dataset['SubjectID'].isin(val_subjects)]
            # Step 2.1: Create features!
            validation_data = self.rv_dataset.create_features(validation_data)

            # print(validation_data.columns)
            print("Size of validation set in Trainer: ", validation_data.shape)

            validation_data = subjective_normalization_dataset(validation_data)
            validation_data = self.rv_dataset.apply_transformation_dataset(validation_data, isTrain=False)
            validation_data = self.rv_dataset.scale_target_dataset(validation_data, isTrain=False)

            assert len(validation_data) > 0, "Validation data is empty."
            validation_sequences = create_sequences(validation_data)

            assert len(validation_sequences) > 0, "Validation sequences are empty."
            validation_features, validation_targets = separate_features_and_targets(
                validation_sequences)
            assert len(validation_features) > 0 and len(
                validation_targets) > 0, "Validation features or targets are empty."

            # print("Size of validation set in Trainer after normalizing and transformation. Number of sequences: ", len(validation_features), " and one sample has shape", validation_features[0].shape)

            # Example usage
            valid_features_tensor, valid_targets_tensor = create_lstm_tensors_dataset(validation_features,
                                                                                      validation_targets)
            valid_loader_0, input_size_0 = create_dataloaders_dataset(valid_features_tensor, valid_targets_tensor,
                                                                      batch_size=batch_size)

            self.valid_loader = valid_loader_0

        if test_index is not None:
            # val_subjects = subject_list[val_index]
            test_subjects = test_index
            print("Test subjects: ", test_subjects)
            test_subjects = [test_subjects] if isinstance(test_subjects, str) else test_subjects
            test_data = self.dataset[self.dataset['SubjectID'].isin(test_subjects)]
            # Step 2.1: Create features!
            test_data = self.rv_dataset.create_features(test_data)
            test_data = subjective_normalization_dataset(test_data)
            test_data = self.rv_dataset.apply_transformation_dataset(test_data, isTrain=False)
            test_data = self.rv_dataset.scale_target_dataset(test_data, isTrain=False)

            assert len(test_data) > 0, "Test data is empty."
            test_sequences = create_sequences(test_data, sequence_length=10)

            assert len(test_sequences) > 0, "Test sequences are empty."
            test_features, test_targets = separate_features_and_targets(test_sequences)
            assert len(test_features) > 0 and len(
                test_targets) > 0, "Validation features or targets are empty."

            # Example usage
            test_features_tensor, test_targets_tensor = create_lstm_tensors_dataset(test_features,
                                                                                    test_targets)
            test_loader_0, input_size_0 = create_dataloaders_dataset(test_features_tensor, test_targets_tensor,
                                                                     batch_size=batch_size)

        return train_loader_0, valid_loader_0, test_loader_0, input_size_0

    def saveActivations(self, userFolder, intermediates, epoch, batch_idx=None):
        # Create a directory to save the activations, if it doesn't exist
        epoch_dir = os.path.join(userFolder, "Activations_figures", f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for layer_name, activations in intermediates.items():
            # Handle the case where activations have more than 2 dimensions
            if activations.dim() > 2:
                # Example: Taking the mean over the sequence dimension
                activations = activations.mean(dim=1)

            activations = activations[:10, :]

            # Convert to numpy for saving or visualization
            activations_np = activations.cpu().detach().numpy()

            # Optional: Visualize activations using matplotlib
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            cax = ax.matshow(activations_np, aspect='auto')
            plt.title(f'Activations: {layer_name}')
            plt.colorbar(cax)

            # Construct a filename that optionally includes batch index
            if batch_idx is not None:
                filename = f"{layer_name}_batch_{batch_idx}.png"
            else:
                filename = f"{layer_name}.png"

            plt.savefig(os.path.join(epoch_dir, filename))
            plt.close()

        # Create a directory to save the activations, if it doesn't exist
        epoch_dir = os.path.join(userFolder, "Activations_tensors", f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for layer_name, activations in intermediates.items():
            # Handle the case where activations have more than 2 dimensions
            filename = f"{layer_name}.pt"
            # Save activations to file
            torch.save(activations, os.path.join(epoch_dir, filename))
