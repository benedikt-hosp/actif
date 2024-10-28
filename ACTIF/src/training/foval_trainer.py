import json
import pickle
import os
from sklearn.model_selection import train_test_split

import keyboard
import torch
import numpy as np
from torch import nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from torch.cuda.amp import autocast, GradScaler

from src.dataset_classes.AbstractDatasetClass import AbstractDatasetClass
from models.foval.FOVAL import FOVAL
from models.foval.utilities import create_optimizer

# torch.backends.cudnn.enabled = False
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU



class FOVALTrainer:
    def __init__(self, config_path, dataset: AbstractDatasetClass, device, save_intermediates_every_epoch):
        """
        Initialize the FOVALTrainer with feature count, dataset object, and model save path.
        """
        self.n_splits = None
        self.current_fold = None
        self.hidden_layer_size = None
        self.dropout_rate = None
        self.fc1_dim = None
        self.patience_counter = 0
        self.early_stopping_threshold = 1.0
        self.patience_limit = 150  # originally 150
        self.hyperparameters = None
        self.feature_count = None # as we need to remove target and subject column
        self.feature_names = None
        self.dataset = dataset  # Dataset object from RobustVisionDataset
        self.config_path = config_path
        self.save_path = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.fold_results = []
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
        self.current_metrics = {}
        self.target_scaler = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = self.dataset.sequence_length
        self.csv_filename = "training_results.csv"
        self.device = device

        self.beta = None
        self.weight_decay = None
        self.learning_rate = None
        self.batch_size = None
        self.l1_lambda = None

        # Flags
        self.save_intermediates_every_epoch = save_intermediates_every_epoch

    def setup(self, feature_count, feature_names):
        self.n_splits = len(self.dataset.subject_list)
        self.feature_count = feature_count  # as we need to remove target and subject column
        self.feature_names = feature_names
        self.load_model_checkpoint(self.config_path)
        self.initialize_model()

    def load_model_checkpoint(self, config_path):
        with open(config_path, 'r') as f:
            hyper_parameters = json.load(f)
        self.hyperparameters = hyper_parameters
        self.batch_size = hyper_parameters['batch_size']
        self.learning_rate = hyper_parameters['learning_rate']
        self.weight_decay = hyper_parameters['weight_decay']
        self.fc1_dim = hyper_parameters['fc1_dim']
        self.dropout_rate = hyper_parameters['dropout_rate']
        self.hidden_layer_size = hyper_parameters['embed_dim']
        self.beta = 0.75

        print("Hyper parameters: ", hyper_parameters)

    def initialize_model(self):
        self.model = FOVAL(input_size=self.feature_count, embed_dim=self.hyperparameters['embed_dim'],
                           fc1_dim=self.hyperparameters['fc1_dim'],
                           dropout_rate=self.hyperparameters['dropout_rate']).to(device)

    def save_activations_and_weights(self, intermediates, filename, file_path):
        save_path_tensors = os.path.join(file_path, f"{filename}_activations.pt")
        save_path_numpy = os.path.join(file_path, f"{filename}_weights.pkl")

        # Separate tensors and numpy arrays
        tensors_dict = {k: v for k, v in intermediates.items() if isinstance(v, torch.Tensor)}
        numpy_dict = {k: v for k, v in intermediates.items() if isinstance(v, np.ndarray)}

        # Save tensors using torch.save
        torch.save(tensors_dict, save_path_tensors)

        # Save numpy arrays using pickle
        with open(save_path_numpy, 'wb') as f:
            pickle.dump(numpy_dict, f)

        print(f"Tensors saved to {save_path_tensors}")
        print(f"NumPy arrays saved to {save_path_numpy}")

    def cross_validate(self, num_epochs=500, loocv=False, num_repeats=5):
        """
        Perform cross-validation on the dataset.
        """

        if loocv:

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            fold_accuracies = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset.subject_list)):
                print(f"\n\nStarting Fold {fold + 1}/{self.n_splits}")

                # Use indices to select subject IDs and convert them to lists
                train_subjects = list(self.dataset.subject_list[train_idx])
                val_subjects = list(self.dataset.subject_list[val_idx])

                # Set the current fold number
                self.current_fold = fold + 1

                fold_mae = self.run_fold(train_subjects, val_subjects, None, num_epochs, loocv=True)
                fold_accuracies.append(fold_mae)
                print(f"Fold {fold + 1} MAE: {fold_mae}")
                average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
                print(f"Average Validation MAE across folds: {average_accuracy}")
                self.reset_metrics()

            # Calculate overall performance on whole dataset
            best_fold = min(fold_accuracies)
            print(f"Best Fold with MAE: {best_fold}")
            average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            print(f"Average Cross-Validation MSE: {average_accuracy}")
            return average_accuracy
        else:

            # Initialize a list to store MAE for each fold
            fold_accuracies = []

            # Set up 5-Fold Cross-Validation
            kf = KFold(n_splits=num_repeats, shuffle=True, random_state=42)

            # Iterate over each fold
            for fold, (train_index, val_index) in enumerate(kf.split(self.dataset.subject_list), 1):
                print(f"\n\nStarting Fold {fold}/{num_repeats}")

                # Get training and validation subjects using the indices provided by KFold
                train_subjects = [self.dataset.subject_list[i] for i in train_index]
                val_subjects = [self.dataset.subject_list[i] for i in val_index]

                # Set the current iteration number as fold
                self.current_fold = fold

                # Perform training and validation for this fold
                fold_mae = self.run_fold(train_subjects, val_subjects, None, num_epochs, loocv=False)
                fold_accuracies.append(fold_mae)

                print(f"Fold {fold} MAE: {fold_mae}")
                self.reset_metrics()

            # Calculate the average performance over the 5 folds
            average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            print(f"Average Validation MAE over 5 folds: {average_accuracy}")

            return fold_accuracies

    def run_fold(self, train_index, val_index=None, test_index=None, num_epochs=10, loocv=False):
        if loocv:
            # Use individual directories per validation participant only when doing LOOCV
            if val_index is not None and len(val_index) > 0:
                validation_participant_name = val_index[0]  # Single participant for LOOCV
                self.save_path = os.path.join("results", validation_participant_name)
            else:
                validation_participant_name = "unknown"
                self.save_path = os.path.join("results", "unknown")
        else:
            # Use a shared save path for standard cross-validation
            self.save_path = os.path.join("results", "current_run")

        # Ensure the directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Prepare data loaders
        self.train_loader, self.valid_loader, self.test_loader, input_size = self.dataset.get_data_loader(
            train_index, val_index, test_index, self.batch_size)

        self.target_scaler = self.dataset.target_scaler
        self.optimizer, self.scheduler = create_optimizer(
            model=self.model, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay)

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            for epoch in range(num_epochs):
                self.train_epoch(epoch)
                if keyboard.is_pressed('q'):
                    goToNextOptimStep = True
                    isBreakLoop = True
                    break  # Exit the outer loop to stop training completely for current subject

                if self.valid_loader:
                    is_last_epoch = (epoch == num_epochs - 1)
                    self.validate_epoch(epoch, is_last_epoch=is_last_epoch)

                self.scheduler.step()

                if self.check_early_stopping(epoch):
                    break
            profiler.step()  # Step profiler at the end of each epoch

        # Save model state dictionary after training
        # SPEED UP: COMMENT
        # self.save_model_state(epoch)
        print("Training finished in epoch ", epoch)
        torch.cuda.empty_cache()
        return self.best_metrics["mae"]

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        scaler = GradScaler()  # Initialize GradScaler for mixed precision

        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss(beta=0.75).to(self.device)

        self.model.train()
        total_samples = 0.0
        total_mae, total_mse, total_smae = 0, 0, 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            # Example forward pass with mixed precision
            with autocast():
                y_pred, _ = self.model(X_batch, return_intermediates=True)
                smae_loss = smae_loss_fn(y_pred, y_batch)
            # Scaled backward pass
            scaler.scale(smae_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # Inverse transform for metric calculation (post-backpropagation)
            y_pred_inv = self.inverse_transform_target(y_pred)
            y_batch_inv = self.inverse_transform_target(y_batch)

            # Accumulate metrics on the original scale
            total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_mse += mse_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_smae += smae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        self.current_metrics["train_mae"] = total_mae / total_samples
        self.current_metrics["train_mse"] = total_mse / total_samples
        self.current_metrics["train_smae"] = total_smae / total_samples

    def validate_epoch(self, epoch, is_last_epoch=False):
        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss().to(self.device)

        self.model.eval()
        total_val_mae, total_val_mse, total_val_smae = 0, 0, 0
        total_val_samples = 0.0
        all_predictions, all_true_values = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.valid_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred, intermediates = self.model(X_batch, return_intermediates=True)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                total_val_mae += mae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_mse += mse_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_smae += smae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_samples += y_batch.size(0)

                all_predictions.append(y_pred.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        # Store metrics for this epoch
        self.current_metrics["val_mae"] = total_val_mae / total_val_samples
        self.current_metrics["val_mse"] = total_val_mse / total_val_samples
        self.current_metrics["val_smae"] = total_val_smae / total_val_samples
        self.current_metrics["val_r2"] = r2_score(np.concatenate(all_true_values), np.concatenate(all_predictions))

        # Save activations and weights based on the condition
        # SPEED UP TRAINING: COMMENT
        # if self.save_intermediates_every_epoch or is_last_epoch:
        #     self.save_activations_and_weights(intermediates, "intermediates", self.save_path)

    def inverse_transform_target(self, y_transformed):
        """
        Apply inverse transformation to the target variable.
        """
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        y_transformed_np = y_transformed.detach().numpy().reshape(-1, 1)
        if self.dataset.target_scaler:
            y_inverse_transformed = self.dataset.target_scaler.inverse_transform(y_transformed_np).flatten()
            return torch.from_numpy(y_inverse_transformed).to(self.device)
        return y_transformed

    def check_early_stopping(self, epoch):
        isBreakLoop = False

        # Check if the current SMAE is better than the best one
        if self.current_metrics["val_mae"] < self.best_metrics["mae"]:
            # Update best SMAE and save model
            self.best_metrics["smae"] = self.current_metrics["val_smae"]
            self.best_metrics["mae"] = self.current_metrics["val_mae"]

            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model_state_dict.pth'))
            print(
                f"Model saved at epoch {epoch} with SMAE {self.current_metrics['val_smae']} and MAE {self.current_metrics['val_mae']}")
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Early stopping logic
        if self.current_metrics[
            "val_mae"] < self.early_stopping_threshold or self.patience_counter > self.patience_limit:
            isBreakLoop = True

        return isBreakLoop
        #
        # isBreakLoop = False
        # # Check validation results
        # if self.current_metrics["val_mae"] < self.best_metrics["mae"]:
        #     self.best_metrics["mae"] = self.current_metrics["val_mae"]
        #
        # if self.current_metrics["val_smae"] < self.best_metrics["smae"]:
        #     self.best_metrics["smae"] = self.current_metrics["val_smae"]
        #
        #     torch.save(self.model.state_dict(), 'results/best_model_state_dict.pth')
        #
        #     self.patience_counter = 0
        #     # self.fold_results = analyzeResiduals(all_predictions_array, all_true_values_array)
        #     # self.all_predictions_array = all_predictions_array
        #     # self.all_true_values_array = all_true_values_array
        #     print(
        #         f'Model saved at epoch {epoch} with validation SMAE {self.best_metrics["smae"]:.6f} and MAE {self.best_metrics["mae"]}\n')
        # else:
        #     self.patience_counter += 1
        #
        # """ 3. Implement early stopping """
        # if self.current_metrics["val_mse"] < self.best_metrics["mae"]:
        #     self.best_metrics["mae"] = self.current_metrics["val_mse"]
        #
        # if self.current_metrics["val_smae"] < self.early_stopping_threshold:
        #     isBreakLoop = True
        #
        # if self.patience_counter > self.patience_limit:
        #     isBreakLoop = True

        return isBreakLoop, self.patience_counter

    def analyzeResiduals(self, predictions, actual_values):
        # Calculate absolute errors
        absolute_errors = np.abs(predictions - actual_values)

        # 1. CALCULATE THE AMOUNT OF GREAT OK AND BAD errors
        """
            Desc: Show how many good, ok, and bad estimations we have (remember to show distribution too) 
            Categorize errors into bins
            bin 1: < 1 cm
            bin 2: < 10 cm
            bin 3: > 20 cm
        """
        bin1 = 1
        bin2 = 10
        bin3 = 20

        errors_under_1cm = np.sum(absolute_errors < bin1) / len(absolute_errors)
        errors_1cm_to_10cm = np.sum((absolute_errors >= bin1) & (absolute_errors <= bin2)) / len(absolute_errors)
        errors_10cm_to_20cm = np.sum((absolute_errors >= bin2) & (absolute_errors < bin3)) / len(absolute_errors)
        errors_above_20cm = np.sum(absolute_errors > bin3) / len(absolute_errors)

        # Print percentages
        print(f"Errors under 1 cm: {errors_under_1cm * 100:.2f}%")
        print(f"Errors between 1 cm and 10 cm: {errors_1cm_to_10cm * 100:.2f}%")
        print(f"Errors between 10 cm and 20 cm: {errors_10cm_to_20cm * 100:.2f}%")
        print(f"Errors above 20 cm: {errors_above_20cm * 100:.2f}%")

        # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
        # 2. Calculate which ranges (in bins of 10 cm )where predicted with wich average error
        """
            Desc: Show which depth values can be predicted the best
            Analyze performance across different depths
            bin size: 10
        """
        bin_size = 10
        # Bin actual values into 10 cm intervals
        bins = np.arange(30, max(actual_values) + bin_size, bin_size)  # Adjust the range as needed
        bin_indices = np.digitize(actual_values, bins)

        # Calculate mean absolute error for each bin
        mean_errors_per_bin = []

        for i in range(1, len(bins)):
            bin_errors = absolute_errors[bin_indices == i]
            mean_error = np.nanmean(bin_errors) if len(bin_errors) > 0 else 0.0
            if mean_error != 0.0:
                mean_errors_per_bin.append(mean_error)

        # Print mean errors per bin
        for i, error in enumerate(mean_errors_per_bin):
            print(f"Depths bin: {bins[i]} to {bins[i + 1]} cm: MAE: {error:.2f} cm")

        # calculate average error between 0.35 and 2 meters
        lower_bound = 35
        upper_bound = 200

        # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
        filtered_errors_for_specific_actual_range = absolute_errors[
            (actual_values >= lower_bound) & (actual_values <= upper_bound)]

        # Calculate the mean of these filtered errors
        average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)

        print(
            f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")

        # calculate average error between 0 and 6 meters
        lower_bound = 0
        upper_bound = 600

        # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
        filtered_errors_for_specific_actual_range = absolute_errors[
            (actual_values >= lower_bound) & (actual_values <= upper_bound)]

        # Calculate the mean of these filtered errors
        average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)

        print(
            f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")

        results = {
            '<1cm': errors_under_1cm * 100,
            '1-10cm': errors_1cm_to_10cm * 100,
            '10-20': errors_10cm_to_20cm * 100,
            '>20': errors_above_20cm * 100,
            'mean_errors_per_bin': {f"{bins[i]} to {bins[i + 1]} cm": error for i, error in
                                    enumerate(mean_errors_per_bin)},
            'average_error_for_2m_range': average_error_for_specific_actual_range
        }
        return results

    def save_model_state(self, epoch):
        """
        Save the model state dictionary after training.
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model_state_dict.pth')))
        model_path = os.path.join(self.save_path, 'optimal_subject_model_state_dict.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Optimal model state dictionary saved at epoch {epoch}.")

    def set_save_path(self, fold_name):
        """
        Set the save path for the current fold.
        """
        # Create a directory for the current fold under results/
        self.save_path = os.path.join("results", fold_name)

        # Ensure the directory exists
        os.makedirs(self.save_path, exist_ok=True)

        print(f"Save path set to: {self.save_path}")

    def reset_metrics(self):
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
        # self.current_metrics = {"val_smae": float('inf'), "mse": float('inf'), "val_mae": float('inf')}
