import gc
import json
import os
from sklearn.model_selection import KFold
import FOVAL_Preprocessor
import torch
import pandas as pd
import numpy as np

from FOVAL_Trainer import FOVAL_Trainer
from FeatureRankingsCreator import FeatureRankingsCreator
from RobustVision_Dataset import RobustVision_Dataset
import warnings
from SimpleLSTM import SimpleLSTM_V2

# ================ Display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

# ================ Device options
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "../model_archive"
os.makedirs(model_save_dir, exist_ok=True)

# ================ Objects options
foval_trainer = FOVAL_Trainer()

# ================ Data split options
fixed_splits = []
n_splits = 25
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


def testModel(model_name, method=None, featureCount=55):
    model_path = os.path.join(model_save_dir, model_name)
    model, hyperparameters = loadModel(model_path, featureCount)

    l1_lambda = hyperparameters['l1_lambda']
    batch_size = hyperparameters['batch_size']
    embed_dim = hyperparameters['embed_dim']
    learning_rate = hyperparameters['learning_rate']
    weight_decay = hyperparameters['weight_decay']
    l1_lambda = l1_lambda
    dropout_rate = hyperparameters['dropout_rate']
    fc1_dim = hyperparameters['fc1_dim']
    beta = 0.75
    all_feature_importances = []

    fold_performance = []
    fold_count = 1
    feature_importances = None

    for fixed_split in fixed_splits:
        train_index, val_index, test_index = fixed_split

        if val_index is None:
            username = f"results/{test_index[0]}"
        else:
            username = f"results/{val_index[0]}"

        isExist = os.path.exists(username)
        if not isExist:
            os.makedirs(username)

        foval_trainer.userFolder = username
        print("User folder is set to: ", foval_trainer.userFolder)

        success, goToNextOptimStep, best_val_mae_fold = foval_trainer.runFold(
            batch_size=batch_size,
            embed_dim=embed_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            dropoutRate=dropout_rate,
            fc1_dim=fc1_dim,
            fold_count=fold_count, n_splits=n_splits,
            train_index=train_index, val_index=val_index,
            test_index=test_index,
            fold_performance=fold_performance, model=model,
            beta=beta, method=method)

        if not success:
            print("Failed")
        else:
            fold_count += 1

    if feature_importances:
        all_feature_importances.append(feature_importances)

    # ============================ CALC Trial performances
    best_fold = min(fold_performance, key=lambda x: x['best_val_mae'])
    print(f"Best Fold: {best_fold['fold']} with MAE: {best_fold['best_val_mae']}")
    average_fold_val_mae = np.mean([f['best_val_smae'] for f in fold_performance])
    print(f"Average Validation MAE across folds: {average_fold_val_mae}")
    # wandb.log({'average_fold_val_mae': average_fold_val_mae})

    return average_fold_val_mae
    # calculate averages of feature importances


def loadModel(model_path, featureCount=54):
    jsonFile = model_path + '.json'
    # modelFile = model_path + ".pt"

    with open(jsonFile, 'r') as f:
        hyperparameters = json.load(f)

    model = SimpleLSTM_V2(input_size=featureCount, embed_dim=hyperparameters['embed_dim'],
                          fc1_dim=hyperparameters['fc1_dim'],
                          dropout_rate=hyperparameters['dropout_rate']).to(device)

    return model, hyperparameters


def get_top_features(importances, percentage):
    # Calculate how many features to include
    num_features = int(len(importances) * percentage)
    # Return the top 'num_features' from the importance list
    top_features = importances[:num_features]

    # Append 'Gt_Depth' and 'SubjectID' to the list of top features
    top_features.extend(['Gt_Depth', 'SubjectID'])

    return top_features


def test_performance_of_ranking_by_method(feature_rankings_creator):
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = {}

    # Baseline performance with full feature set
    all_features = FOVAL_Preprocessor.selected_features
    feature_count = len(all_features) - 2
    foval_trainer.rv_dataset.remaining_features = all_features
    full_feature_performance = testModel(model_name=modelName, featureCount=feature_count)
    results['Baseline'] = full_feature_performance
    with open("ACTIF_evaluation_results.txt", "a") as file:
        file.write(f"Baseline Performance: {full_feature_performance}\n")

    # Loop through each method
    for method, importances in feature_rankings_creator.feature_importance_results.items():
        results[method] = {}
        # Test each percentage of features
        for percent in percentages:
            top_features = get_top_features(importances, percent)
            feature_count = len(top_features) - 2
            foval_trainer.rv_dataset.remaining_features = top_features
            performance = testModel(model_name=modelName, featureCount=feature_count)
            results[method][f'{int(percent * 100)}%'] = performance

            result_line = f'Method: {method}, Percent: {percent * 100}%, Performance: {performance}\n'
            with open("ACTIF_evaluation_results.txt", "a") as file:
                file.write(result_line)
            print(result_line)

            # Manually release memory
            del top_features
            gc.collect()

    print("All results:", results)
    # Optionally, write all results at once if needed
    with open("ACTIF_evaluation_results.txt", "a") as file:
        file.write(str(results))


if __name__ == '__main__':

    """
        Setup Model
    """
    sequence_length = 10  # Define your sequence length
    foval_trainer.sequence_length = sequence_length
    rv_dataset = RobustVision_Dataset(sequence_length=sequence_length)  # Initialize the data processor
    aggregated_data, _ = rv_dataset.read_and_aggregate_data()  # Step 1: Read and aggregate data and clean
    foval_trainer.dataset = aggregated_data
    foval_trainer.rv_dataset = rv_dataset

    # # Assuming 'subject_list' contains the list of unique subjects
    subject_list = foval_trainer.dataset['SubjectID'].unique()

    # Create data splits for LOOCV
    test_subjects_list = subject_list
    for train_subjects, val_subjects in kf.split(test_subjects_list):
        fixed_splits.append((subject_list[train_subjects], subject_list[val_subjects], None))

    # Define which model to load
    modelName = "lstm_BestSMAE_12.22_AvgSMAE_16.88"

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # ACTIF Creation:
    # Calculate feature importance ranking for all methods
    # collect sorted ranking list, memory usage, and computation speed
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    model_path = os.path.join(model_save_dir, modelName)
    _, hyperparameters = loadModel(model_path)
    fmv = FeatureRankingsCreator(hyperparameters, foval_trainer, subject_list)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # ACTIF Evaluation:
    # Runs all feature importance lists with 10-60 % of their top features and compares their runtime
    # memory consumption and performance
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    test_performance_of_ranking_by_method(fmv)
