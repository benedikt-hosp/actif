import gc
import json

import numpy as np
import torch
import pandas as pd
import os
import sys

from src.dataset_classes.giw_dataset import GIWDataset
from src.dataset_classes.TuftsDataset import TuftsDataset
from src.models.FOVAL.foval_trainer import FOVALTrainer
from src.models.FOVAL.foval_preprocessor import input_features
import warnings
from src.dataset_classes.robustVision_dataset import RobustVisionDataset

# ================ Display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

# ================ Randomization seed
np.random.seed(42)
torch.manual_seed(42)

# ================ Device options
device = torch.device("cuda")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "models"
os.makedirs(model_save_dir, exist_ok=True)
BASE_DIR = './'
MODEL = "FOVAL"
DATASET_NAME = "TUFTS"  # "ROBUSTVISION"  "TUFTS" "GIW"


def build_paths(base_dir):
    print("BASE_DIR is ", base_dir)
    paths = {"model_save_dir": os.path.join(base_dir, "model_archive"),
             "results_dir": os.path.join(base_dir, "results"),
             "data_base": os.path.join(base_dir, "data", "input"),
             "model_path": os.path.join(base_dir, "models", MODEL, "config", MODEL),
             "config_path": os.path.join(base_dir, "models", MODEL, "config", MODEL)}

    paths["data_dir"] = os.path.join(paths["data_base"], DATASET_NAME)
    paths["results_folder_path"] = os.path.join(paths["results_dir"], MODEL, DATASET_NAME, "FeaturesRankings_Creation")
    paths["evaluation_metrics_save_path"] = os.path.join(paths["results_dir"], MODEL, DATASET_NAME)

    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return paths


def get_top_features(importances, percentage):
    # Calculate how many features to include
    num_features = int(len(importances) * percentage)
    # Return the top 'num_features' from the importance list
    top_features = importances[:num_features]

    # Append 'Gt_Depth' and 'SubjectID' to the list of top features
    top_features.extend(['Gt_Depth', 'SubjectID'])

    return top_features


def test_list(feature_list, modelName, dataset, methodName, trainer, save_path, num_repetitions=2):
    percentages = [0.1,  0.4]  # , 0.2, 0.3]
    results = {}
    list_name = f"{modelName}_{dataset.name}_{methodName}"
    results[list_name] = {}

    print("Loc 1: ")
    save_path = os.path.join(save_path, "performance_evaluation_metrics.txt")

    # Dynamically create the header based on the number of repetitions
    run_columns = ", ".join([f"Run {i + 1}" for i in range(num_repetitions)])
    header = f"Method, Percent, {run_columns}, Mean MAE, Standard Deviation\n"


    for percent in percentages:
        top_features = get_top_features(feature_list, percent)
        feature_count = len(top_features) - 2  # Adjust based on your specific needs
        remaining_features = top_features
        print(f"3. Evaluating top {int(percent * 100)}% features.")

        # Assign the top features to the trainer
        trainer.dataset = dataset
        dataset.current_features = remaining_features
        dataset.feature_count = feature_count
        dataset.load_data()

        print("Remaining features: ", remaining_features)
        trainer.setup(features=remaining_features)

        # Perform cross-validation and get the performance results for each run
        fold_accuracies = trainer.cross_validate(num_epochs=200)

        # Calculate mean and standard deviation
        mean_performance = np.mean(fold_accuracies)
        std_dev_performance = np.std(fold_accuracies, ddof=1)

        # Create the result line
        runs_values = ", ".join([f"{accuracy:.4f}" for accuracy in fold_accuracies])
        result_line = (f"{list_name}, {percent * 100:.0f}%, "
                       f"{runs_values}, {mean_performance:.4f}, {std_dev_performance:.4f}\n")
        # result_line = "{}\t{:.0f}%\t{}\t{:.4f}\t{:.4f}\n".format(
        #     list_name, percent * 100, "\t".join(map(str, runs_values)), mean_performance, std_dev_performance
        # )

        try:
            if not os.path.exists(save_path):
                with open(save_path, "w") as file:
                    file.write(header + "\n")  # Header
            with open(save_path, "a") as file:
                file.write(result_line)
        except Exception as e:
            print(f"Error writing to file: {e}")

        print(result_line)
        # Manually release memory
        del top_features
        # torch.cuda.empty_cache()
        gc.collect()

    return results


def test_baseline_model(trainer, modelName, dataset, outputFolder, num_repetitions=10):
    results = {}
    top_features = input_features
    feature_count = len(top_features) - 2  # Adjust based on your specific needs
    remaining_features = top_features
    print(f"START: Evaluating BASELINE Model.")

    # Assign the top features to the trainer
    trainer.dataset = dataset
    dataset.current_features = remaining_features
    dataset.load_data()

    trainer.setup(feature_count=feature_count, feature_names=remaining_features)

    # Perform cross-validation and get the performance results for each run
    full_feature_performance = trainer.cross_validate(num_epochs=500, loocv=False, num_repeats=num_repetitions)

    results['Baseline'] = full_feature_performance

    # Save baseline results
    print("Baseline saved to ", outputFolder)
    with open(outputFolder, "a") as file:
        file.write(f"Baseline Performance of {modelName} on dataset {dataset.name}: {full_feature_performance}\n")

    print(f"Baseline Performance of {modelName} on dataset {dataset.name}: {full_feature_performance}\n")
    return full_feature_performance


def getFeatureList(path):
    # Read the CSV file using pandas
    data = pd.read_csv(path)
    print("Data ", data)
    # Sort theDataFrame by the second column (index 1)
    sorted_data = data.sort_values(by=data.columns[1])

    # Extract the first column (index 0) as a sorted list
    sorted_first_column = sorted_data[data.columns[0]].tolist()

    # Display the sorted list of the first column
    print(sorted_first_column)

    return sorted_first_column


if __name__ == '__main__':
    # Parameterize MODEL and DATASET folders
    paths = build_paths(BASE_DIR)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 1. Define Dataset
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    datasetName = DATASET_NAME
    if DATASET_NAME == "ROBUSTVISION":
        # num_repetitions = 25  # Define the number of repetitions for 80/20 splits
        dataset = RobustVisionDataset(data_dir=paths["data_dir"], sequence_length=10)
        dataset.name = DATASET_NAME
        dataset.load_data()
        num_repetitions = len(dataset.subject_list)
    elif DATASET_NAME == "GIW":
        dataset = GIWDataset(data_dir=paths["data_dir"], trial_name="T4_tea_making")
        dataset.name = DATASET_NAME
        dataset.load_data()
        num_repetitions = len(dataset.subject_list)
    elif DATASET_NAME == "TUFTS":
        dataset = TuftsDataset(data_dir=paths["data_dir"])
        dataset.name = DATASET_NAME
        dataset.load_data()
        num_repetitions = len(dataset.subject_list)
    else:
        print("No dataset chosen.")

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 2. Define Model
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    if MODEL == "FOVAL":
        trainer = FOVALTrainer(config_path=paths["config_path"], dataset=dataset, device=device,
                               feature_names=input_features, save_intermediates_every_epoch=False)
        # trainer.setup()

    # 1. Baseline performance evaluation
    print(f" 1. Testing baseline {MODEL} on dataset {datasetName}")
    # baseline_performance = test_baseline_model(trainer, MODEL, dataset, paths["save_path"], num_repetitions)

    # 2. Loop over all feature lists (CSV files) and evaluate
    for file_name in os.listdir(paths["results_folder_path"]):
        if file_name.endswith(".csv"):
            file_path = os.path.join(paths["results_folder_path"], file_name)
            # print("File path is", file_path)
            method = file_name.replace('.csv', '')  # Extract method name from file
            current_feature_list = getFeatureList(file_path)
            print("features are", current_feature_list)

            print(f" 2. Evaluating feature list: {method}")
            test_list(feature_list=current_feature_list, dataset=dataset, modelName=MODEL, methodName=method,
                      trainer=trainer, save_path=paths["evaluation_metrics_save_path"], num_repetitions=num_repetitions)




# Ablation inv: 10%  7.063386930672503 tufts