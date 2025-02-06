from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler, Normalizer, Binarizer, FunctionTransformer
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


class AbstractDatasetClass(ABC):
    def __init__(self, data_dir, sequence_length):
        self.input_data = None
        self.subject_list = None
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.best_transformers = None
        self.minDepth = 0.35
        self.maxDepth = 3
        self.subject_scaler = RobustScaler()
        self.feature_scaler = None
        self.target_scaler = None
        self.target_column_name = 'Gt_Depth'
        self.subject_id_column = 'SubjectID'
        self.multiplicator = 100

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

    @abstractmethod
    def load_data(self):
        """
        Load data from the specified directory. Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def create_features(self, data_in):
        """
        Create and generate features from the input data. Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def normalize_data(self, data):
        """
        Normalize the data globally and subject-wise. Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def calculate_transformations_for_features(self, data_in):
        """
        Calculate the best transformations for the features. Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def apply_transformations_on_features(self, data_in):
        """
        Apply the transformations on the features for validation or test data.
        """
        pass

    @abstractmethod
    def scale_target(self, data_in, isTrain=False):
        """
        Scale the target column ('Gt_Depth') in the dataset.
        """
        pass

    @abstractmethod
    def scale_features(self, data_in, isTrain=True):
        """
        Scale the features in the dataset.
        """
        pass

    @abstractmethod
    def create_sequences(self, df):
        """
        Create sequences for time-series analysis.
        """
        pass

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

    def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=100):
        """
        Create and return data loaders for training, validation, and testing datasets.
        """
        train_loader = self.prepare_loader(train_index, batch_size, is_train=True)
        val_loader = self.prepare_loader(val_index, batch_size, is_train=False) if val_index is not None else None
        # test_loader = self.prepare_loader(test_index, batch_size, is_train=False) if test_index is not None else None

        input_size = train_loader.dataset[0][0].shape[1]  # Assuming the first dimension is batch_size

        return train_loader, val_loader, input_size

    @abstractmethod
    def prepare_loader(self, subject_index, batch_size, is_train=False):
        """
        Prepare and return a data loader for the specified subjects.
        """
        pass

    def get_data(self):
        return self.input_data
