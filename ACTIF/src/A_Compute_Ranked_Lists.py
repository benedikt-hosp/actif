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


if __name__ == '__main__':

    # 1. Modelle:
    #   FOVAL:  GIW, Robust Vision, Tufts
    #   XXX: 1 Datensatz
    #   YYY: 1 Datensatz

    # 2. Sensitivity analysis: V1:Memory Efficient, V2: Fast Execution, V3: High Precision
    # deeplift:     zero, random, mean input baselines
    # nisp: 1. No mixed precision, full accumulation of activations
    #       2. Half-precision FP16
    #       3. Limiting accumulation to only specific layers
    # IntGrad:
    # SHAP:
    # Ablation:
    # Shuffle:

    # 3. Vergleiche:
    # Für jedes Modell, jeden Datensatz, jede Methode, mit allen Sensitivity Parameter, mit jeder ACTIF Variante
    # Modelle: 3
    # Datensätze: 3
    # Methode: 10
    # Actif Varianten: 4
    # Sensitivitätsparameter: bis zu 3 (gesamt 60 parameterisierte Methoden)

    # Methoden: 60 Methoden (alle Parametervarianten, ACTIF Varianten) pro Datensatz
    # Datensätze: 3
    # Gesamt: 180 Methoden = Ranked Lists

    # Auswertung: Für jede Methode mean über alle Datensätze und innerhalb Datensatz vergleichen
    #   - Memory Consumption
    #   - Computing Time
    #   - Performance

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 1. Define Dataset
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    dataset = RobustVisionDataset(data_dir="../data/input/robustvision/", sequence_length=10)
    dataset.load_data()

    modelName = "Foval"
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 3. Create Ranked Lists
    # ACTIF Creation: Calculate feature importance ranking for all methods collect sorted ranking list, memory usage, and computation speed
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    fmv = FeatureRankingsCreator(modelName=modelName, datasetName='robustvision', dataset=dataset)
    fmv.process_methods()



'''
LOGs:
14.10.2024:
    - Next steps are 
     a) evaluate performance of the created lists: 
     FIRST, because I am not sure how well ACTIF turns out with DeepLift and NISP, only if ACTIF shows good performance in comparison
     b) apply methods on other dataset
     c) pick another model with 2 datasets and do the same
     
 08.11.2024:
    - Status: Performance on RV dataset with 10% evaluated
    - Steps: 
        - Check if other version of deepActif can be reproduced
        - check 20% / 30%
        - apply methods on other dataset
        - pick another model with 2 datasets and do the same
'''