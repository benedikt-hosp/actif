import csv
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import wandb

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from SimpleLSTM import SimpleLSTM_V2

# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visualizeSingleSubjectPredictions(userfolder, all_true_values_array, all_predictions_array):
    # Save the arrays
    np.save(os.path.join(userfolder, 'subject_true_values_last_valid.npy'), all_true_values_array)
    np.save(os.path.join(userfolder, 'subject_predicted_values_last_valid.npy'), all_predictions_array)

    scatter_color = '#648c94'  # light blue for scatter plot
    line_color = '#993768'  # purple for the line
    bar_color = '#3c6c7c'  # not used here but set up for consistency

    # Setting up the plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-darkgrid')  # For gray background and grid

    # Scatter plot with label for legend
    plt.scatter(all_true_values_array, all_predictions_array, color=scatter_color, edgecolor='black', alpha=1.0, s=20,
                label='Errors')

    # Diagonal reference line with label for legend
    plt.plot([all_true_values_array.min(), all_true_values_array.max()],
             [all_true_values_array.min(), all_true_values_array.max()],
             'k--', lw=2, color=line_color, label='Optimal correlation')

    # Customizing the plot
    plt.title('Predicted vs. True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)

    # Including the legend
    plt.legend()
    plt.savefig(f"{userfolder}/ScatterTruevsPred.png", format='png', dpi=300, bbox_inches='tight')

    scatter_color = '#648c94'  # light blue for scatter plot
    line_color = '#993768'  # purple for the line
    bar_color = '#3c6c7c'

    # Combine, sort by true values, and split
    combined = np.array(sorted(zip(all_true_values_array, all_predictions_array), key=lambda x: x[0]))
    sorted_true_values = combined[:, 0]
    sorted_predicted_values = combined[:, 1]

    # Calculate mean and standard deviation for sorted predicted values
    mean_predictions = np.mean(sorted_predicted_values)
    std_dev_predictions = np.std(sorted_predicted_values)

    # Identify outliers
    outlier_indices = np.where(abs(sorted_predicted_values - mean_predictions) > 2 * std_dev_predictions)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_true_values, color='#507858', label='True Values', linewidth=2)
    plt.plot(sorted_predicted_values, color='#993768', linestyle='--', label='Predicted Values', linewidth=2)

    # Highlight outliers in sorted data
    plt.scatter(outlier_indices, sorted_predicted_values[outlier_indices], color='orange', label='Outliers')

    # Adding shaded area for standard deviation
    plt.fill_between(range(len(sorted_predicted_values)),
                     sorted_predicted_values - std_dev_predictions,
                     sorted_predicted_values + std_dev_predictions,
                     color=scatter_color, alpha=0.2)

    plt.title('Sorted True vs. Predicted Values')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"{userfolder}/Distribution_conf.png", format='png', dpi=300, bbox_inches='tight')

    # plt.show()

    # TODO: DISTRIBUTION OF PREDICTION ERRORS
    errors = all_predictions_array - all_true_values_array
    print(f"Mean is {np.nanmean(errors)} and std dev is {np.nanstd(errors)}")

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=25, color=bar_color, edgecolor='lightgray')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"{userfolder}/Distribution.png", format='png', dpi=300, bbox_inches='tight')

    # plt.show()


def visualizeErrors(averaged_results):
    name = "average_errors.npy"

    path = os.path.join("results", name)
    np.save(path, averaged_results)

    if averaged_results is None:
        print("No data provided")
        return

    # TODO: Error Distribution Across Folds
    # Error Distribution Bar Plot
    plt.figure(figsize=(10, 6))
    plt.bar(averaged_results['error_distribution'].keys(), averaged_results['error_distribution'].values(),
            color='#3c6c7c', edgecolor='lightgray')
    plt.xlabel('Error Categories')
    plt.ylabel('Percentage')
    plt.title('Error Distribution Across Folds')

    plt.savefig(f"results/ErrorDistributionAcrossFolds.png", format='png', dpi=300, bbox_inches='tight')

    # plt.show()

    # TODO: Mean Errors Per Depth Bin Line Plot
    plt.figure(figsize=(14, 6))
    keys = averaged_results['mean_errors_per_bin'].keys()
    values = averaged_results['mean_errors_per_bin'].values()
    plt.plot(keys, values, marker='o', linestyle='-', color='#993768')
    plt.xticks(rotation=45)
    plt.xlabel('Depth Bins')
    plt.ylabel('Mean Absolute Error (cm)')
    plt.title('Mean Errors Per Depth Bin Across Folds')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/MeanErrorDepthBins.png", format='png', dpi=300, bbox_inches='tight')

    # plt.show()


def define_model_and_optim(input_size, embed_dim, dropoutRate, learning_rate, weight_decay, fc1_dim, model=None):
    # Set a seed before initializing your model
    seed_everything(seed=42)

    # Instantiate the model
    if model is None:
        model = SimpleLSTM_V2(input_size=input_size, embed_dim=embed_dim, dropout_rate=dropoutRate, fc1_dim=fc1_dim).to(
            device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=learning_rate)  # 100

    return model, optimizer, scheduler


def create_lstm_tensors_dataset(X, y):
    # Check for None or empty inputs
    assert X is not None and len(X) > 0, "X is empty or None"
    assert y is not None and len(y) > 0, "y is empty or None"

    # Convert training features
    features = np.array([sequence.values for sequence in X])
    print(f"features shape: {features.shape}")
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Convert training targets
    targets_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    print(
        f"Training dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} and {len(y)} labels")

    return features_tensor, targets_tensor


def create_dataloaders_dataset(features_tensor, targets_tensor, batch_size):
    train_dataset = TensorDataset(features_tensor, targets_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True,  num_workers=2, pin_memory=True, prefetch_factor=2)

    #  Assuming X_train has shape[num_samples, sequence_length, num_features]
    input_size = features_tensor.shape[2]  # Number of features per time step
    print("Input size is: ", input_size)

    return train_loader, input_size


def analyzeResiduals(predictions, actual_values):
    # Calculate absolute errors
    absolute_errors = np.abs(predictions - actual_values)

    # 1. CATEGORIZE ERRORS. CALCULATE THE AMOUNT OF GREAT, OK AND BAD errors
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
    # 2. Best working range. Calculate which ranges (in bins of 10 cm ) where predicted with wich average error
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
    # wandb.log({'val_2Meter_MAE': average_error_for_specific_actual_range})

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


def print_results(iteration, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay,
                  fc1_dim,
                  avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2,
                  best_train_mse, best_train_mae, best_train_smae, avg_val_mse, avg_val_rmse, avg_val_mae,
                  avg_val_smae, avg_val_r2, best_val_mse, best_val_mae, best_val_smae):
    """ 4. Print current run average values"""
    print(f"Run: {iteration}:\n"
          f" TRAINING: MSE:", avg_train_mse, " \tRMSE:", avg_train_rmse, " \tMAE:", avg_train_mae, " \tHuber: ",
          avg_train_smae, " \tR-squared:", avg_train_r2,
          " \tBest train MSE:", best_train_mse, "\tBest train MAE:", best_train_mae, "\t Best train Huber:",
          best_train_smae,
          " \nVALIDATION: \tMSE:", avg_val_mse, " \tRMSE:", avg_val_rmse, " \tMAE:", avg_val_mae, " \tHuber: ",
          avg_val_smae, " \tR-squared:", avg_val_r2,
          " \tBest val MSE:", best_val_mse, " \tBest Val MAE:", best_val_mae, " \tBest val Huber:", best_val_smae,
          "\n",
          " Batch size: ", batch_size, " \tDropout: ", dropoutRate, " \tembed_dim :", embed_dim, " \tfc1_dim :",
          fc1_dim,
          " \tLearning rate: ", learning_rate, "\tWeight decay: ", weight_decay, "\tL1: ", l1_lambda)
    #
    # saveToTSV(iteration, avg_val_mse, batch_size, embed_dim, dropoutRate, l1_lambda,
    #           learning_rate, weight_decay, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae,
    #           avg_train_r2, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, best_val_mse,
    #           best_val_mae, best_val_smae)


def saveToTSV(csv_filename, iteration, target, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate,
              weight_decay,
              avg_train_mse,
              avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2, avg_val_mse, avg_val_rmse, avg_val_mae,
              avg_val_smae, avg_val_r2, best_val_mse, best_val_mae, best_val_smae):
    # Open the CSV file in write mode and initialize the writer
    with open(csv_filename, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the data to the CSV file
        writer.writerow([
            iteration, target, batch_size, dropoutRate, embed_dim,
            l1_lambda, learning_rate, weight_decay, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae,
            avg_train_r2,
            avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, best_val_mse, best_val_mae,
            best_val_smae
        ])
