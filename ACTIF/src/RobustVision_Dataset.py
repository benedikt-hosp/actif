import string

import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler, QuantileTransformer, \
    StandardScaler, MinMaxScaler, FunctionTransformer, Binarizer
import warnings
from FOVAL_Preprocessor import detect_and_remove_outliers, binData, createFeatures, \
    detect_and_remove_outliers_in_features_iqr

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_cleaning(df):
    # # Remove rows where all elements are NaN
    # df = df.dropna(how='all')
    #
    # df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.dropna().copy()
    #
    # df2 = df[df['Gt_Depth'] > 0.35]
    # df2 = df2[df2['Gt_Depth'] <= 3]
    #
    # df2["Gt_Depth"] = df2["Gt_Depth"].multiply(100)
    #
    # df2 = df2.reset_index(drop=True)
    #
    # # Detect outliers
    # df3 = detect_and_remove_outliers(df2, window_size=10, threshold=30)  # 5, 100
    # df4 = detect_and_remove_outliers_in_features_iqr(df3)
    # df4 = binData(df4)

    # Remove rows where all elements are NaN
    df = df.dropna(how='all')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()

    # df2 = df[df['GT depth'] > 0.1]
    df2 = df[df['Gt_Depth'] > 0.35]
    df2 = df2[df2['Gt_Depth'] <= 3]

    df2["Gt_Depth"] = df2["Gt_Depth"].multiply(100)

    df2 = df2.reset_index(drop=True)

    # Detect outliers
    df3 = detect_and_remove_outliers(df2, window_size=5, threshold=10)  # 5, 100
    # df4 = self.detect_and_remove_outliers_in_features(df2)
    df4 = detect_and_remove_outliers_in_features_iqr(df3)

    df4 = binData(df4, False)
    # self.plotFeatures(df4)
    return df4


def create_sequences(df, sequence_length=10):
    sequences = []
    # Group data by subject
    grouped_data = df.groupby('SubjectID')
    for subj_id, group in grouped_data:
        for i in range(len(group) - sequence_length):
            seq_features = group.iloc[i:i + sequence_length].drop(columns=['Gt_Depth', 'SubjectID'])
            seq_target = group.iloc[i + sequence_length]['Gt_Depth']
            sequences.append((seq_features, seq_target, subj_id))
    return sequences


class RobustVision_Dataset:
    pd.option_context('mode.use_inf_as_na', True)

    def __init__(self, sequence_length):
        self.best_transformers = None
        self.feature_scaler = None
        self.target_scaler = None
        self.train_targets = None
        self.valid_features = None
        self.valid_targets = None
        self.train_features = None
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

        self.use_minmax = True
        self.use_standard_scaler = False
        self.use_robust_scaler = False
        self.use_quantile_transformer = False
        self.use_power_transformer = False
        self.use_max_abs_scaler = False
        self.sequence_length = sequence_length
        self.remaining_features = []

    def clean_column_names(self, df):
        printable = set(string.printable)
        df.columns = [''.join(filter(lambda x: x in printable, col)) for col in df.columns]

    def create_features(self, data_in):
        print("CREATE FEATURES CALLED ===============================================")
        data_in = createFeatures(data_in, isGIW=False)

        # If we want to test a subset of features we will do so in feature importance checking and
        # then we will set remaining_features to a subset. If not, we should skip this step
        if self.remaining_features:
            data_in = data_in[self.remaining_features]
        print("Remaining cols: ", data_in.columns)
        return data_in

    # 1.
    def read_and_aggregate_data(self):
        print("=============================================================================\n\n")
        print("\tTraining and evaluating regression model on SFB Robust Vision Dataset\n\n")
        print("=============================================================================")

        data_dir = "../data/Subject_25"

        subject_ids = []
        all_data = []

        # 0. Read in files
        for subj_folder in os.listdir(data_dir):
            subj_path = os.path.join(data_dir, subj_folder)

            print(subj_path)

            if os.path.exists(subj_path):
                depthCalib_path = os.path.join(subj_path, "depthCalibration.csv")
                if os.path.exists(depthCalib_path):
                    df = pd.read_csv(depthCalib_path, delimiter="\t")
                    df.rename(columns=lambda x: x.strip().replace(' ', '_').replace('_', '_').title().replace('Gaze_',
                                                                                                              'Gaze_'), inplace=True)

                    # Extract and typecast specific columns
                    starting_columns = ['Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                                        'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X',
                                        'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z',
                                        'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z',
                                        'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z']

                    # self.clean_column_names(df_depthEval)
                    df = df[starting_columns]
                    for col in starting_columns:
                        df[col] = df[col].astype(float)

                    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
                    df = base_cleaning(df)

                    df['SubjectID'] = subj_folder
                    all_data.append(df)

                    # Add the subject ID to the list
                    subject_ids.append(subj_folder)

        print("Finished reading subjects and saved sequences.")
        # Combine all data into a single dataframe
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data, None

    def apply_transformation(self, trainset, validset):
        trainset_transformed = self.transform_and_visualize(trainset)
        validationset_transformed = self.apply_transformations_to_validation(validset)

        return trainset_transformed, validationset_transformed

    def apply_transformation_dataset(self, dataset, isTrain=False):
        if isTrain:
            dataset_transformed = self.transform_and_visualize(dataset)
        else:
            dataset_transformed = self.apply_transformations_to_validation(dataset)

        return dataset_transformed

    def transform_and_visualize(self, data_in):
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

    def apply_transformations_to_validation(self, validation_data):
        transformed_validation_data = validation_data.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is not None:
                if column == "Gt_Depth":
                    transformed_validation_data[column] = validation_data[[column]]
                elif column == "SubjectID":
                    continue
                else:
                    # Apply the transformation using the fitted transformer object
                    transformed_column = transformer.transform(validation_data[[column]])
                    transformed_validation_data[column] = transformed_column.squeeze()

        return transformed_validation_data

    def scale_target_dataset(self, data_in, isTrain=False):

        if isTrain:
            scaler2 = None
            if self.use_minmax:
                print("BHO Scaler: Using minmax scaler")
                scaler2 = MinMaxScaler(feature_range=(0, 1000))
                # scaler2 = MinMaxScaler()

            if self.use_standard_scaler:
                print("BHO Scaler: Using Standard scaler")
                scaler2 = StandardScaler()

            if self.use_robust_scaler:
                print("BHO Scaler: Using robust scaler")
                scaler2 = RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)

            if self.use_quantile_transformer:
                print("BHO Scaler: Using quantile transformer")
                scaler2 = QuantileTransformer(output_distribution='normal')

            if self.use_power_transformer:
                print("BHO Scaler: Using power transformer")
                scaler2 = PowerTransformer(method='yeo-johnson')

            if self.use_max_abs_scaler:
                print("BHO Scaler: Using max abs scaler")
                scaler2 = MaxAbsScaler()

            # Initialize a separate scaler for GT_depth
            self.target_scaler = scaler2

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

    def scale_data(self, training_data, validation_data):
        scaler = None
        scaler2 = None
        if self.use_minmax:
            print("BHO Scaler: Using minmax scaler")
            scaler = MinMaxScaler(feature_range=(0, 1000))
            scaler2 = MinMaxScaler(feature_range=(0, 1000))

        if self.use_standard_scaler:
            print("BHO Scaler: Using Standard scaler")
            scaler = StandardScaler()
            scaler2 = StandardScaler()

        if self.use_robust_scaler:
            print("BHO Scaler: Using robust scaler")
            scaler = RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)
            scaler2 = RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)

        if self.use_quantile_transformer:
            print("BHO Scaler: Using quantile transformer")
            scaler = QuantileTransformer(output_distribution='normal')
            scaler2 = QuantileTransformer(output_distribution='normal')

        if self.use_power_transformer:
            print("BHO Scaler: Using power transformer")
            scaler = PowerTransformer(method='yeo-johnson')
            scaler2 = PowerTransformer(method='yeo-johnson')

        if self.use_max_abs_scaler:
            print("BHO Scaler: Using max abs scaler")
            scaler = MaxAbsScaler()
            scaler2 = MaxAbsScaler()

        # Initialize a separate scaler for GT_depth

        # Extract GT_depth before scaling and reshape for scaler compatibility
        gt_depth_train = training_data['Gt_Depth'].values.reshape(-1, 1)
        gt_depth_validation = validation_data['Gt_Depth'].values.reshape(-1, 1)

        subject_id_train = training_data['SubjectID']
        subject_id_validation = validation_data['SubjectID']

        # Drop the excluded columns from the datasets
        training_data = training_data.drop(columns=['Gt_Depth', 'SubjectID'])
        validation_data = validation_data.drop(columns=['Gt_Depth', 'SubjectID'])

        # Assign the target scaler to an instance variable for later inverse transformation
        self.target_scaler = scaler2

        # If a feature scaler is set, fit and transform the training data, and transform the validation data
        if scaler is not None:
            training_data_scaled = scaler.fit_transform(training_data)
            validation_data_scaled = scaler.transform(validation_data)

            # Convert the scaled arrays back to pandas DataFrames
            training_data = pd.DataFrame(training_data_scaled, columns=training_data.columns)
            validation_data = pd.DataFrame(validation_data_scaled, columns=validation_data.columns)

            gt_depth_train_scaled = scaler2.fit_transform(gt_depth_train)
            gt_depth_validation_scaled = scaler2.transform(gt_depth_validation)

            # Re-attach the excluded columns
            training_data['Gt_Depth'] = gt_depth_train_scaled.ravel()
            validation_data['Gt_Depth'] = gt_depth_validation_scaled.ravel()

            training_data['SubjectID'] = subject_id_train.reset_index(drop=True)
            validation_data['SubjectID'] = subject_id_validation.reset_index(drop=True)

        else:
            # Convert the scaled arrays back to pandas DataFrames
            training_data = pd.DataFrame(training_data, columns=training_data.columns)
            validation_data = pd.DataFrame(validation_data, columns=validation_data.columns)

            # Re-attach the excluded columns
            training_data['Gt_Depth'] = gt_depth_train.ravel()
            validation_data['Gt_Depth'] = gt_depth_validation.ravel()
            training_data['SubjectID'] = subject_id_train.reset_index(drop=True)
            validation_data['SubjectID'] = subject_id_validation.reset_index(drop=True)

        return training_data, validation_data
