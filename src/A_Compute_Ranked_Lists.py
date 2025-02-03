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
             "config_path": os.path.join(base_dir, "models", MODEL, "config", DATASET_NAME, ".json")}

    paths["data_dir"] = os.path.join(paths["data_base"], DATASET_NAME)
    paths["results_folder_path"] = os.path.join(paths["results_dir"], MODEL, DATASET_NAME, "FeaturesRankings_Creation")
    paths["evaluation_save_path"] = os.path.join(paths["results_dir"], MODEL, DATASET_NAME, "ACTIF_evaluation_results.txt")

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

'''
LOGs:
14.10.2024:
    - Next steps are 
     a) evaluate performance of the created lists: 
     FIRST, because I am not sure how well ACTIF turns out with DeepLift and NISP, only if ACTIF shows good performance in comparison
     b) apply methods on other dataset
     c) pick another model with 2 datasets and do the same
     
     LRP, IG, SHAP, Ablation, Shuffle, ACTIF, DeepLift, NISP, GradCAM, GradCAM++, Saliency Maps, Attention Maps.

01.02.2025:
    
    # Ziel: Feature Importance Ranking of neural network

    # 1. Modelle:
    #   FOVAL
    #   XXX: 1 Datensatz
    #   YYY: 1 Datensatz

    # 2. Methoden:
    # 2.1 Modellagnostische Methoden (Blackbox-Modelle interpretieren)
    # Deeplift:     zero, random, mean baselines
    # NISP:
    # IntGrad:
    # SHAP (KernelSHAP, TreeSHAP)
    # Ablation:
    # Permutation:
    # NEU
    # LIME / Anchors
    # LRP
    # Occlusion Sensitivity


    # 2.2 Gradienten-basierte (für neuronale Netze)
    # Vanilla Gradients (Saliency Maps) -> BesseR: Guided Backpropagation
    # Smoothgrad
    # DeepLIFT

    # 2.3 Methoden für bestimmete Architekture
    # 2.3.1 CNNs
    # GRAD-CAM / GRAD-CAM++ / Score-CAM
    # LRP
    # Deep Taylor Decomposition
    # Feature Visualization (Neuron Activation Maximization, NAM
    # Attention Maps für Vision Transformer (ViTs)

    # 2.4 LSTM/RNN Zeitreihne/NLP
    # Gradient-based Attribution (wie bei CNNs)
    # Attention Weight Visualization
    # Input Cell State Contribution (z. B. für LSTMs & GRUs)
    # Temporal Importance Ranking (z. B. mit Shapley Values auf Zeitfenstern)

    # 2.5 Methoden für Transformer (NLP & Vision)
	# Attention Rollout: Summiert mehrschichtige Attention-Gewichte für besser verständliche Heatmaps.
	# Attention Attribution (z. B. LIME auf Attention-Scores)
	# Gradient-based Methoden (Integrated Gradients, SHAP, DeepLIFT für Transformer)

    # 2.6 Regelbasierte Methoden
    # Antwortet nicht „Warum ist Feature X wichtig?“ sondern „Welche Datenpunkte haben die Vorhersage beeinflusst?“
    # Counterfactual Explanations (CFEs)
    # 	•	Zeigt minimale Änderungen an den Eingabedaten, die die Vorhersage ändern würden.
    # 	•	Beispiel: „Wenn Einkommen +500€ wäre → Kreditzusage!“
    # 	•	Prototypen & Kritische Beispiele (ProtoDash, Deep Prototypes)
    # 	•	Zeigt typische Beispiele, die das Modell repräsentieren.
    # 	•	K-Nearest Neighbors (KNN) für Interpretierbarkeit
    # 	•	„Welche Trainingsdaten sind der aktuellen Eingabe am ähnlichsten?“
    # 	•	Influence Functions
    # 	•	Analysiert, welche Trainingsbeispiele eine Vorhersage am stärksten beeinflusst haben.
    # 	•	Adversarial Explanations
    # 	•	Identifiziert minimale Eingangsänderungen, die eine falsche Klassifikation verursachen.

    # 2.7 Uncertainty-aware Methoden (Feature Ranking mit Unsicherheitsschätzung)
    # 🔹 Besonders für deine Arbeit relevant!
    # 	•	Bayesian Deep Learning (z. B. Monte Carlo Dropout)
    # 	•	Deep Ensembles für Unsicherheitsquantifizierung
    # 	•	Gaussian Processes über Feature-Attributionen
    # 	•	Variance-based Feature Importance (Wie stabil ist das Ranking über verschiedene Mini-Batches?)
    # 	•	Bayesian Feature Ranking Updating (Dynamische Anpassung des Rankings über Zeit)

    # 3. Vergleiche:
    # Für jedes Modell, jeden Datensatz, jede Methode, mit allen Sensitivity Parameter, mit jeder ACTIF Variante
    # Modelle: 3
    # Datensätze: 3
    # Methode: 10
    # Actif Varianten: 4
    # Sensitivitätsparameter: bis zu 3 (gesamt 60 parameterisierte Methoden)
    
    
    # TODO Suggestions for Improvement:
    # 	1. Include additional baselines like LIME and LRP to provide a more comprehensive benchmark.
    # 	2. Expand dataset diversity by including datasets from different sources but for the same task
    #    	(e.g., another biometric dataset)
    #          + TUFTS
    #          + GIW
    #       and exploring domains like finance, speech processing, and industrial monitoring.
    #           # ?
    # 	3. Test additional architectures to show DeepACTIF’s flexibility beyond LSTMs.
    #       - TCNs
    #       - Transformers,
    # 	4. Provide qualitative explainability comparisons by visualizing feature importance rankings and checking consistency across different methods.
    # 	5. Analyze robustness to model and data variations, including sensitivity to LSTM size, dataset noise, and sequence length.
    # 	6. Investigate potential bias in INV by checking whether penalizing high-variability features removes important signals.

    # Auswertung: Für jede Methode mean über alle Datensätze und innerhalb Datensatz vergleichen
    #   - Memory Consumption
    #   - Computing Time
    #   - Performance     
     '''
