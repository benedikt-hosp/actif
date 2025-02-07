import gc
import json
import os
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
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "../model_archive"
os.makedirs(model_save_dir, exist_ok=True)
currentFeatureList = []


def get_top_features(importances, percentage):
    # Calculate how many features to include
    num_features = int(len(importances) * percentage)
    # Return the top 'num_features' from the importance list
    top_features = importances[:num_features]

    # Append 'Gt_Depth' and 'SubjectID' to the list of top features
    top_features.extend(['Gt_Depth', 'SubjectID'])

    return top_features


def test_list(feature_list, modelName, dataset, methodName, trainer, save_path):
    percentages = [0.1,
                   0.2,
                   0.3,
                   0.4,
                   0.5,
                   # 0.6
                   ]
    results = {}
    list_name = modelName + '_' + dataset.name + '_' + methodName

    results[list_name] = {}

    for percent in percentages:
        top_features = get_top_features(feature_list, percent)
        feature_count = len(top_features) - 2  # Adjust based on your specific needs
        remaining_features = top_features
        print(f" 3. with top {percent} % features. ")

        trainer.feature_names = remaining_features
        trainer.dataset = dataset
        performance = trainer.cross_validate(num_epochs=500)
        results[list_name][f'{int(percent * 100)}%'] = performance

        result_line = f'Method: {list_name}, Percent: {percent * 100}%, Performance: {performance}\n'
        with open(save_path, "a") as file:
            file.write(result_line)
        print(result_line)

        # Manually release memory
        del top_features
        gc.collect()

    print("All results:", results)
    # Optionally, write all results at once if needed
    with open("ACTIF_evaluation_results.txt", "a") as file:
        file.write(str(results))


def test_baseline_model(trainer, modelName, dataset):
    results = {}
    # Baseline performance with full feature set
    all_features = input_features
    feature_count = len(all_features) - 2
    trainer.feature_names = all_features
    trainer.dataset = dataset
    full_feature_performance = trainer.cross_validate()
    results['Baseline'] = full_feature_performance
    with open("ACTIF_evaluation_results.txt", "a") as file:
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
    # Setup Model
    modelName = "Foval"
    datasetName = "robustvision"

    dataset = RobustVisionDataset(data_dir="../data/input/robustvision/", sequence_length=10)
    dataset.load_data()
    model, hyperparameters = loadFOVALModel(model_path="../models/foval/config/foval")
    trainer = FOVALTrainer(config_path="../models/foval/config/foval.json", dataset=dataset, device=device,
                           feature_names=input_features, save_intermediates_every_epoch=False)
    trainer.setup()

    # 1. Baseline performance evaluation
    print(f" 1. Testing baseline {modelName} on dataset {datasetName}")
    # baseline_performance = test_baseline_model(trainer, modelName, dataset)

    # Loop over all feature lists (CSV files)
    folder_path = f'../results/{modelName}/{dataset.name}/FeaturesRankings_Creation'
    save_path = f'../results/{modelName}/{dataset.name}/ACTIF_evaluation_results.txt'



    for file_name in reversed(os.listdir(folder_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            method = file_name.replace('.csv', '')  # Method name from the file name
            current_feature_list = getFeatureList(file_path)
            print(f" 2.Testing list {method}")
            test_list(feature_list=current_feature_list, dataset=dataset, modelName=modelName, methodName=method,
                      trainer=trainer, save_path=save_path)
