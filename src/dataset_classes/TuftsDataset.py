import math
import pickle

import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler,
    QuantileTransformer, StandardScaler, MinMaxScaler,
    FunctionTransformer, Binarizer
)
import warnings

from src.dataset_classes.AbstractDatasetClass import AbstractDatasetClass
from src.models.FOVAL.foval_preprocessor import remove_outliers_in_labels, binData, createFeatures, \
    detect_and_remove_outliers_in_features_iqr, clean_data, global_normalization, subject_wise_normalization, \
    separate_features_and_targets
from src.models.FOVAL.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TuftsDataset(AbstractDatasetClass):
    def __init__(self, data_dir, test_split_size=10):
        """
        Initialize the GIWDataset class.

        :param data_dir: Directory containing the trial folders.
        :param trial_name: The specific trial to load (e.g., 'T1_Indoor_Walk').
        """
        super().__init__(data_dir, sequence_length=10)
        self.data_dir = data_dir
        self.name = None
        self.test_split_size = test_split_size
        self.input_data = None
        self.target_column_name = 'Gt_Depth'
        self.features = None
        self.input_data = None
        self.subject_list = None
        self.current_features = None
        self.sequence_length = 10  # sequence_length
        self.data_dir = data_dir
        self.best_transformers = None
        self.isGIW= False # für mixed muss das auf True stehen sonst False
        self.minDepth = 0.35  # in meter
        self.maxDepth = 3
        self.current_features = None
        self.subject_scaler = RobustScaler()  # or any other scaler
        self.feature_scaler = None
        self.target_scaler = None
        self.target_column_name = 'Gt_Depth'
        self.subject_id_column = 'SubjectID'
        self.multiplicator = 100  # to convert from cm to m
        self.transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
            'RobustScaler': RobustScaler,
            'QuantileTransformer-Normal': lambda: QuantileTransformer(output_distribution='normal'),
            'QuantileTransformer-Uniform': lambda: QuantileTransformer(output_distribution='uniform'),
            'PowerTransformer-YeoJohnson': lambda: PowerTransformer(method='yeo-johnson'),
            'PowerTransformer-BoxCox': lambda: PowerTransformer(method='box-cox'),
            'Normalizer': Normalizer,
            'Binarizer': lambda threshold=0.0: Binarizer(threshold=threshold),
            'FunctionTransformer-logp1p': lambda func=np.log1p: FunctionTransformer(func),
            'FunctionTransformer-rec': lambda func=np.reciprocal: FunctionTransformer(func),
            'FunctionTransformer-sqrt': lambda func=np.sqrt: FunctionTransformer(func),
        }

        self.scaler_config = {
            'use_minmax': True,
            'use_standard_scaler': False,
            'use_robust_scaler': False,
            'use_quantile_transformer': False,
            'use_power_transformer': False,
            'use_max_abs_scaler': False
        }

        self.scaler_config_features = {
            'use_minmax': False,  # avg 56
            'use_standard_scaler': False,  # avg 15
            'use_robust_scaler': False,  # avg 15.5
            'use_quantile_transformer': False,  # avg 14.4
            'use_power_transformer': False,  # avg 15.2
            'use_max_abs_scaler': False  # avg 10.79
            # none                               # avg
        }

    # 1.

    def load_data(self):

        all_data = []

        # Path to the trial directory containing parsed CSV files
        trial_dir = os.path.join(self.data_dir,'parsed_to_csv')

        # Loop through all CSV files in the trial directory
        for csv_file in os.listdir(trial_dir):
            csv_path = os.path.join(trial_dir, csv_file)

            if os.path.exists(csv_path) and csv_file.endswith('.csv'):
                print(f"Processing file: {csv_file}")

                # Load the CSV file
                df = pd.read_csv(csv_path, delimiter="\t")

                # Rename and select relevant columns
                df.rename(columns=lambda x: x.strip().replace(' ', '_').title(), inplace=True)
                expected_columns = [
                    'Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                    'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y',
                    'World_Gaze_Direction_R_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Y',
                    'World_Gaze_Origin_R_Z',
                    'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Y', 'World_Gaze_Origin_L_Z'
                ]

                # Keep only the necessary columns
                df = df[expected_columns]
                subject_id = csv_file.split('_')[0] + '_' + csv_file.split('_')[1]
                print("Subject is ", subject_id)

                df2 = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
                df3 = remove_outliers_in_labels(df2, window_size=5, threshold=10,
                                                target_column_name=self.target_column_name)
                df4 = detect_and_remove_outliers_in_features_iqr(df3)
                df5 = binData(df4, False)

                # Extract SubjectID from the filename
                # subject_id = self.extract_subject_id_from_filename(csv_file)
                subject_id = csv_file.split('_')[0] + '_' + csv_file.split('_')[1]

                df5['SubjectID'] = subject_id

                # Append the dataframe to the list
                all_data.append(df5)

        # Concatenate all data into a single dataframe
        if all_data:
            self.input_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded {len(all_data)} files with {self.input_data.shape[0]} rows.")
        else:
            print(f"No CSV files found in {trial_dir}.")
        self.subject_list = self.input_data['SubjectID'].unique()

        return self.input_data

    def getIPD(self, row):
        """
        Calculate the IPD based on the ground truth depth and vergence angle.

        Args:
            row: A pandas Series row containing the vergence angle and the known focused depth.

        Returns:
            The calculated IPD for the subject.
        """
        vergenceAngle = row['Vergence_Angle']
        focusedDepth = row['Gt_Depth']  # Ground truth depth (in cm)

        # Ensure the vergence angle and depth are numeric
        vergenceAngle = pd.to_numeric(vergenceAngle, errors='coerce')
        focusedDepth = pd.to_numeric(focusedDepth, errors='coerce')

        if pd.isna(vergenceAngle) or pd.isna(focusedDepth):
            return np.nan  # Invalid or missing values

        try:
            # Calculate IPD using the ground truth depth and vergence angle
            ipd = 2 * focusedDepth * math.tan(math.radians(vergenceAngle) / 2)
        except ValueError as e:
            print(f"Error in calculating IPD: {e}")
            ipd = np.nan  # Return NaN in case of calculation errors

        return ipd * 10

    # 2.
    def create_features(self, data_in):
        """
        Generate features from the input data.

        :param data_in: Input dataframe.
        :return: Dataframe with additional features.
        @param data_in:
        @return:
        """
        data_in = createFeatures(data_in, isGIW=self.isGIW)

        return data_in

    # 3.
    def normalize_data(self, data):

        # Apply global normalization first
        data_set_in = global_normalization(data)

        # Then proceed with the existing subject-wise normalization
        unique_subjects = data_set_in['SubjectID'].unique()

        # Choose your scaler for subject-wise normalization
        subject_scaler = RobustScaler()

        # Apply subject-wise normalization
        dataset_in_normalized = subject_wise_normalization(data_set_in, unique_subjects, subject_scaler)

        return dataset_in_normalized

    # 4. A (Traininig data)
    def calculate_transformations_for_features(self, data_in):
        best_transformers = {}
        transformed_data = data_in.copy()
        ideal_skew = 0.0
        ideal_kurt = 3.0

        for column in data_in.columns:
            if column == "Gt_Depth" or column == "SubjectID":
                continue

            # print(f"Processing column: {column}")
            original_skew = skew(data_in[column])
            original_kurt = kurtosis(data_in[column], fisher=False)  # Pearson's definition

            best_transform = None
            best_transform_name = ""
            min_skew_diff = float('inf')

            for name, transformer_class in self.transformers.items():
                transformer = transformer_class()  # Create a new object for each transformer
                try:
                    data_transformed = transformer.fit_transform(data_in[[column]])
                    current_skew = skew(data_transformed)[0]
                    current_kurt = kurtosis(data_transformed, fisher=False)[0]

                    # Calculate the distance from the ideal distribution characteristics
                    dist = np.sqrt((current_skew - ideal_skew) ** 2 + (current_kurt - ideal_kurt) ** 2)

                    # If this transformer is the best so far, store it
                    if dist < min_skew_diff:
                        min_skew_diff = dist
                        best_transform = transformer
                        best_transform_name = name

                except ValueError as e:  # Handle failed transformations, e.g., Box-Cox with negative values
                    # print(f"Transformation failed for {name} on column {column}: {e}")
                    continue

            best_transformers[column] = (best_transform_name, best_transform)

            # Transform the column in the dataset
            if best_transform:
                transformed_column = best_transform.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        self.best_transformers = best_transformers
        return transformed_data

    # 4. B ( Validation/ test data)
    def apply_transformations_on_features(self, data_in):
        transformed_validation_data = data_in.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is not None:
                if column == "Gt_Depth":
                    transformed_validation_data[column] = data_in[[column]]
                elif column == "SubjectID":
                    continue
                else:
                    # Apply the transformation using the fitted transformer object
                    transformed_column = transformer.transform(data_in[[column]])
                    transformed_validation_data[column] = transformed_column.squeeze()

        return transformed_validation_data

    # 5.
    def scale_target(self, data_in, isTrain=False):

        if isTrain:
            self.target_scaler = self.select_scaler(self.scaler_config)
            # Extract GT_depth before scaling and reshape for scaler compatibility
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.fit_transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()
        else:
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()

        return data_in

    # 6.
    def scale_features(self, data_in, isTrain=True):
        """
        Scale the features in the training and validation datasets using the provided scaler.

        :param data_in: Dataframe with training data.
        :return: Scaled dataframe.
        @param isTrain:
        """

        # target_column = data_in[self.target_column_name].values.reshape(-1, 1)
        # subject_id_column = data_in[self.subject_id_column]
        # data = data_in.drop(columns=[self.target_column_name, self.subject_id_column])
        #
        # # Fit the scaler on the training data
        # if isTrain:
        #     self.feature_scaler = self.select_scaler(self.scaler_config_features)
        #     # Fit the scaler only if it's not already fitted (e.g., during validation)
        #     data_scaled = self.feature_scaler.fit_transform(data)
        # else:
        #     data_scaled = self.feature_scaler.transform(data)
        #
        # data_out = pd.DataFrame(data_scaled, columns=data.columns)
        # data_out[self.target_column_name] = target_column.ravel()
        # data_out[self.subject_id_column] = subject_id_column.reset_index(drop=True)

        return data_in

    # 6.
    def create_sequences(self, df):
        """
        Create sequences of data for time-series analysis.

        :param df: Dataframe with the data to sequence.
        :return: List of sequences, where each sequence is a tuple of (features, target, subject_id).
        """
        sequences = []
        grouped_data = df.groupby('SubjectID')
        for subj_id, group in grouped_data:
            for i in range(len(group) - self.sequence_length):
                seq_features = group.iloc[i:i + self.sequence_length].drop(columns=['Gt_Depth', 'SubjectID'])
                seq_target = group.iloc[i + self.sequence_length]['Gt_Depth']
                sequences.append((seq_features, seq_target, subj_id))
        return sequences

    # Utilities
    def select_scaler(self, config):
        """Select the scaler based on the configuration provided."""
        if config['use_minmax']:
            return MinMaxScaler(feature_range=(0, 1000))
        if config['use_standard_scaler']:
            return StandardScaler()
        if config['use_robust_scaler']:
            return RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)
        if config['use_quantile_transformer']:
            return QuantileTransformer(output_distribution='normal')
        if config['use_power_transformer']:
            return PowerTransformer(method='yeo-johnson')
        if config['use_max_abs_scaler']:
            return MaxAbsScaler()
        return None

    def get_data(self):
        return self.input_data

    def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=100):
        """
        Create and return data loaders for training, validation, and testing datasets.

        :param train_index: Indices for training subjects.
        :param val_index: Indices for validation subjects (optional).
        :param test_index: Indices for test subjects (optional).
        :param batch_size: Batch size for the data loaders.
        :return: Data loaders for training, validation, and testing datasets, and the input size.
        """
        train_loader = self.prepare_loader(train_index, batch_size, is_train=True)
        val_loader = self.prepare_loader(val_index, batch_size, is_train=False) if val_index is not None else None
        test_loader = self.prepare_loader(test_index, batch_size, is_train=False) if test_index is not None else None

        input_size = train_loader.dataset[0][0].shape[1]  # Assuming the first dimension is batch_size
        print("Input size: ", input_size)

        return train_loader, val_loader, input_size

    def prepare_loader(self, subject_index, batch_size, is_train=False):
        subjects = subject_index if isinstance(subject_index, list) else [subject_index]
        # print(f"Preparing data for subjects: {subjects}")
        data = self.input_data[self.input_data['SubjectID'].isin(subjects)]
        # print(data.head)
        # if is_train:
        #     data.to_csv('checkpoint_raw_1.csv')

        # Check if the data is empty before proceeding
        if data.empty:
            raise ValueError(f"No data found for subjects: {subjects}")

        # Feature creation and normalization
        data = self.create_features(data)

        print("Features in Dataaset are ", self.current_features)
        data = data[self.current_features]
        data = self.normalize_data(data)

        # Apply transformations if necessary
        if is_train:
            #     data.to_csv('checkpoint_normalized_3.csv')

            data = self.calculate_transformations_for_features(data)
            # data.to_csv('checkpoint_transformed_4.csv')

        else:
            data = self.apply_transformations_on_features(data)

        # Scale features and target (transform only using the fitted scaler)
        # data = self.scale_features(data, isTrain=is_train)
        data = self.scale_target(data, isTrain=is_train)
        # if is_train:
        #     data.to_csv('checkpoint_scaled_5.csv')

        # Generate sequences
        sequences = self.create_sequences(data)
        features, targets = separate_features_and_targets(sequences)

        # Convert to tensors and create data loader
        features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets)
        data_loader = create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)

        # if is_train:
        #     # Assuming your list is called sequences
        #     with open('train_sequences_new.pkl', 'wb') as f:
        #         pickle.dump(sequences, f)

        # sequences.to_pickle("train_sequences_new.pkl")

        return data_loader
