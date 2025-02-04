import os
import torch
import pandas as pd
import numpy as np

from src.training.foval_trainer import FOVALTrainer
from FeatureRankingsCreator import FeatureRankingsCreator
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
device = torch.device("mps")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "models"
os.makedirs(model_save_dir, exist_ok=True)
BASE_DIR = '../'
MODEL = "FOVAL"
DATASET_NAME = "ROBUSTVISION"  # "GIW"  "TUFTS"


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


if __name__ == '__main__':
    # Parameterize MODEL and DATASET folders
    paths = build_paths(BASE_DIR)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 1. Define Dataset
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    datasetName = DATASET_NAME
    if DATASET_NAME == "ROBUSTVISION":
        num_repetitions = 25  # Define the number of repetitions for 80/20 splits
        dataset = RobustVisionDataset(data_dir=paths["data_dir"], sequence_length=10)
        dataset.load_data()

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 2. Define Model
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    if MODEL == "FOVAL":
        trainer = FOVALTrainer(config_path=paths["config_path"], dataset=dataset, device=device,
                               save_intermediates_every_epoch=False)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 3. Create Ranked Lists
    # ACTIF Creation: Calculate feature importance ranking for all methods collect sorted ranking list, memory usage, and computation speed
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    fmv = FeatureRankingsCreator(modelName=MODEL, datasetName=DATASET_NAME, dataset=dataset, trainer=trainer, paths=paths)
    fmv.process_methods()
