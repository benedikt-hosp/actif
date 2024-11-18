import gc
import json
import os

import numpy as np
import torch
import pandas as pd

from models.foval.FOVAL import FOVAL
from models.foval.foval_preprocessor import input_features
from src.dataset_classes.robustVision_dataset import RobustVisionDataset
import warnings
from src.training.foval_trainer import FOVALTrainer

# ================ Display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

# ================ Device options
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "../model_archive"
os.makedirs(model_save_dir, exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)


def get_top_features(importances, percentage):
    # Calculate how many features to include
    num_features = int(len(importances) * percentage)
    # Return the top 'num_features' from the importance list
    top_features = importances[:num_features]

    # Append 'Gt_Depth' and 'SubjectID' to the list of top features
    top_features.extend(['Gt_Depth', 'SubjectID'])

    return top_features


def test_list(feature_list, modelName, dataset, methodName, trainer, save_path, num_repetitions=10):
    percentages = [0.1, 0.2, 0.3]  # , 0.2, 0.3]
    results = {}
    list_name = f"{modelName}_{dataset.name}_{methodName}"
    results[list_name] = {}

    # Dynamically create the header based on the number of repetitions
    run_columns = ", ".join([f"Run {i + 1}" for i in range(num_repetitions)])
    header = f"Method, Percent, {run_columns}, Mean MAE, Standard Deviation\n"

    # # Write the header row to the file
    # with open(save_path, "w") as file:
    #     file.write(header)

    for percent in percentages:
        top_features = get_top_features(feature_list, percent)
        feature_count = len(top_features) - 2  # Adjust based on your specific needs
        remaining_features = top_features
        print(f"3. Evaluating top {int(percent * 100)}% features.")

        # Assign the top features to the trainer
        trainer.dataset = dataset
        dataset.current_features = remaining_features
        dataset.load_data()

        trainer.setup(feature_count=feature_count, feature_names=remaining_features)

        # Perform cross-validation and get the performance results for each run
        fold_accuracies = trainer.cross_validate(num_epochs=300, loocv=False, num_repeats=num_repetitions)

        # Calculate the mean and standard deviation of the MAE values
        mean_performance = np.mean(fold_accuracies)
        std_dev_performance = np.std(fold_accuracies, ddof=1)

        # Dynamically create the result line based on the number of runs
        runs_values = ", ".join([f"{fold_accuracies[i]:.4f}" for i in range(num_repetitions)])
        result_line = (
            f"{list_name}, {percent * 100}%, "
            f"{runs_values}, {mean_performance:.4f}, {std_dev_performance:.4f}\n"
        )

        # Save the results to the file
        with open(save_path, "a") as file:
            file.write(result_line)

        print(result_line)

        # Store results in the dictionary
        # results[list_name][f'{int(percent * 100)}%'] = {
        #     "runs": performance,
        #     "mean": mean_performance,
        #     "std_dev": std_dev_performance
        # }

        # Manually release memory
        del top_features
        gc.collect()

    return results

    # Save final results for this feature list
    # DOPPELT:
    # print("All results:", results)
    # with open(save_path, "a") as file:
    #     file.write(str(results))


def test_baseline_model(trainer, modelName, dataset, outputFolder, num_repetitions=10):
    results = {}
    all_features = input_features
    feature_count = len(all_features) - 2
    trainer.feature_names = all_features
    trainer.dataset = dataset

    # Perform cross-validation for the baseline
    full_feature_performance = trainer.cross_validate(num_epochs=500, loocv=False, num_repeats=num_repetitions)
    results['Baseline'] = full_feature_performance

    # Save baseline results

    outfile = os.path.join(outputFolder, "ACTIF_evaluation_results.txt")
    print("Baseline saved to ", outputFolder)
    with open(outputFolder, "a") as file:
        file.write(f"Baseline Performance of {modelName} on dataset {dataset.name}: {full_feature_performance}\n")

    print(f"Baseline Performance of {modelName} on dataset {dataset.name}: {full_feature_performance}\n")
    return full_feature_performance


def loadFOVALModel(model_path, featureCount=54):
    jsonFile = model_path + '.json'

    # Print the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    with open(jsonFile, 'r') as f:
        hyperparameters = json.load(f)

    model = FOVAL(input_size=featureCount, embed_dim=hyperparameters['embed_dim'],
                  fc1_dim=hyperparameters['fc1_dim'],
                  dropout_rate=hyperparameters['dropout_rate']).to(device)

    return model, hyperparameters


def getFeatureList(path):
    # Read the CSV file using pandas
    data = pd.read_csv(path)
    # Sort the DataFrame by the second column (index 1)
    sorted_data = data.sort_values(by=data.columns[1])

    # Extract the first column (index 0) as a sorted list
    sorted_first_column = sorted_data[data.columns[0]].tolist()

    # Display the sorted list of the first column
    print(sorted_first_column)

    return sorted_first_column


if __name__ == '__main__':

    import sys

    print("Python version:", sys.version)

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)  # CUDA version PyTorch was built with
    print("cuDNN version:", torch.backends.cudnn.version())  # cuDNN version
    print("Is CUDA available:", torch.cuda.is_available())  # Check if GPU is available


    # Setup Model
    modelName = "Foval"
    datasetName = "robustvision"
    num_repetitions = 10  # Define the number of repetitions for 80/20 splits

    dataset = RobustVisionDataset(data_dir="../data/input/robustvision/", sequence_length=10)

    # Load the model
    model, hyperparameters = loadFOVALModel(model_path="../models/foval/config/foval")

    # Initialize the trainer
    trainer = FOVALTrainer(config_path="../models/foval/config/foval.json", dataset=dataset, device=device,
                           save_intermediates_every_epoch=False)

    # Define paths
    folder_path = f'../results/{modelName}/{dataset.name}/FeaturesRankings_Creation'
    save_path = f'../results/{modelName}/{dataset.name}/ACTIF_evaluation_results.txt'

    # 1. Baseline performance evaluation
    print(f" 1. Testing baseline {modelName} on dataset {datasetName}")
    # baseline_performance = test_baseline_model(trainer, modelName, dataset, save_path, num_repetitions)

    # 2. Loop over all feature lists (CSV files) and evaluate
    for file_name in reversed(os.listdir(folder_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            method = file_name.replace('.csv', '')  # Extract method name from file
            current_feature_list = getFeatureList(file_path)

            print(f" 2. Evaluating feature list: {method}")
            test_list(feature_list=current_feature_list, dataset=dataset, modelName=modelName, methodName=method,
                      trainer=trainer, save_path=save_path, num_repetitions=num_repetitions)
