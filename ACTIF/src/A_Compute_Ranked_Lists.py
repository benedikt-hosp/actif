import json
import os
import torch
import pandas as pd

from FeatureRankingsCreator import FeatureRankingsCreator

import warnings

from models.foval.FOVAL import FOVAL
from src.dataset_classes.robustVision_dataset import RobustVisionDataset

# ================ Display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

# ================ Device options
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "../models"
os.makedirs(model_save_dir, exist_ok=True)


def loadFOVALModel(model_path, featureCount=54):
    jsonFile = model_path + '.json'

    with open(jsonFile, 'r') as f:
        hyperparameters = json.load(f)

    model = FOVAL(input_size=featureCount, embed_dim=hyperparameters['embed_dim'],
                  fc1_dim=hyperparameters['fc1_dim'],
                  dropout_rate=hyperparameters['dropout_rate']).to(device)

    return model, hyperparameters


if __name__ == '__main__':

    # 1. Modell:
    #   FOVAL:  GIW, Robust Vision, Tufts
    #   XXX: 1 Datensatz
    #   YYY : 1 Datensatz

    # 2. Sensitivity analysis:
    # deeplift: zero, random, mean input baselines
    # nisp: 1. No mixed precision, full accumulation of activations
    #       2. Half-precision FP16
    #       3. Limiting accumulation to only specific layers
    # IntGrad:
    # SHAP:
    # Ablation:
    # Shuffle:

    # 3.
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 1. Define Dataset
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    dataset = RobustVisionDataset(data_dir="../data/input/robustvision/", sequence_length=10)
    dataset.load_data()

    model, hyperparameters = loadFOVALModel(model_path="../models/foval/config/foval")
    raw = True
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 3. Create Ranked Lists
    # ACTIF Creation: Calculate feature importance ranking for all methods collect sorted ranking list, memory usage, and computation speed
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    fmv = FeatureRankingsCreator(modelName='Foval', model=model, hyperparameters=hyperparameters, datasetName='robustvision', dataset=dataset, raw=raw)
    fmv.process_methods()
