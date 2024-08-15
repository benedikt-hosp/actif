# ACTIF: Feature Ranking and Model Evaluation

This project implements a pipeline for ranking features using various methods and evaluating model performance based on these rankings. The feature rankings are generated using different importance methods, and the model performance is evaluated with subsets of top features.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Detailed Code Description](#detailed-code-description)
6. [License](#license)
7. [Contributing](#contributing)
8. [Contact](#contact)

## 1. Introduction

This project is designed to facilitate the creation of feature rankings and evaluate the performance of a deep learning model using subsets of these ranked features. The core functionalities include:

- **Feature Rankings Creation:** Uses different methods to rank features based on their importance.
- **Model Evaluation:** Evaluates the model's performance by training on the top-ranked features and comparing it to the baseline performance with all features.

The project is built on top of PyTorch and includes utilities for handling datasets, training models, and computing feature importance.

## 2.Installation

### Prerequisites

- Anaconda/Miniconda
- Python 3.x

### Environment Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/benedikt-hosp/actif.git
   cd actif
   ```

2. **Setup the conda environment:**
    ```
    conda env create -f environment.yml
    conda activate actif_env
    ``` 
    
### Directory Setup
actif/
│

├── data/ 

├── notebooks/         

├── src/               

├── model_archive/     

├── results/           

└── environment.yml    

### 3. Usage
## Running the Feature Ranking and Model Evaluation
#### Generate Feature Rankings:
To generate feature rankings using various methods, run the script src/FeatureRankingsCreator.py. This script will compute feature importances and save them in the 	FeaturesRankings_Creation directory.
```
python src/FeatureRankingsCreator.py
```
    
#### Evaluate Model Performance:

After generating the feature rankings, you can evaluate the model's performance using these rankings. Run the script src/test_performance_of_ranking_by_method.py:
```
python src/test_performance_of_ranking_by_method.py
```

This will evaluate the model's performance on subsets of features ranked by different methods.

## 4. Project Structure

data/: 		Contains data files or datasets.

notebooks/:	Jupyter notebooks for exploratory data analysis and experiments.

src/:		Source code, including feature ranking methods, model training, and evaluation scripts.

model_archive/: Directory to store pre-trained or saved models.

results/: 	Directory where output results and logs are stored.

environment.yml:Conda environment setup file.



## 5. Detailed Code Description
#### FeatureRankingsCreator.py: 
This script handles the generation of feature rankings using various methods such as ACTIF, SHAP, Captum, and more.
test_performance_of_ranking_by_method.py: This script evaluates the model performance using subsets of features ranked by different methods.

#### RobustVision_Dataset.py:
Handles the loading, processing, and management of the Robust Vision dataset.

#### FOVAL_Trainer.py:
Contains the model training logic and evaluation functions.

#### SimpleLSTM_V2.py:
Defines the Simple LSTM model used in this project.

## 6. License
This project is licensed under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You are free to share and adapt the material as long as appropriate credit is given.

## 7. Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bug fixes, features, or improvements.







