import math
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.utils import resample
from joblib import Parallel, delayed, Memory

# original
input_features = [
    'SubjectID',
    'Gt_Depth',
    # INPUT FEATURE
    'World_Gaze_Direction_R_X',
    'World_Gaze_Direction_R_Y',
    'World_Gaze_Direction_R_Z',
    'World_Gaze_Direction_L_X',
    'World_Gaze_Direction_L_Y',
    'World_Gaze_Direction_L_Z',
    # 'World_Gaze_Origin_R_X',
    # 'World_Gaze_Origin_R_Z',
    # 'World_Gaze_Origin_L_X',
    # 'World_Gaze_Origin_L_Z',
    # basic computations
    'Vergence_Angle',
    'Normalized_Vergence_Angle',
    'Vergence_Depth',
    'Cosine_Angles',
    'Delta_Gaze_X',
    'Delta_Gaze_Y',
    'Delta_Gaze_Z',
    'Gaze_Vector_Angle',  # EVA?
    'Gaze_Point_Euclidean_Distance',
    'Directional_Magnitude_R',
    'Directional_Magnitude_L',
    'Directional_Magnitude_Ratio',
    'Gaze_Direction_X_Ratio',
    'Gaze_Direction_Y_Ratio',
    'Gaze_Direction_Z_Ratio',
    'Ratio_Delta_Gaze_XY',
    # 'Velocity_X',
    # 'Acceleration_X',
    'Gaze_Direction_Angle',
    'Relative_Change_Vergence_Angle',
    'Angular_Difference_Gaze_Directions',
    'Ratio_Directional_Magnitude',
    'Gaze_Point_Distance',
    'Gaze_Point_Depth_Difference',
    'Ratio_World_Gaze_Direction_X',
    'Ratio_World_Gaze_Direction_Y',
    'Ratio_World_Gaze_Direction_Z',
    'Velocity_Gaze_Direction_R_X',
    'Acceleration_Gaze_Direction_R_X',
    'Angular_Difference_X',
]

minDepth = 0.35
maxDepth = 3
memory = Memory("cache_dir", verbose=0)


def detect_and_remove_outliers_in_features_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1 * IQR))

    # Filter out the rows with outliers
    new_df = df[~mask.any(axis=1)]
    return new_df


def remove_outliers_in_labels(df, window_size, threshold, target_column_name):
    # Check if 'Gt_Depth' column exists
    if 'Gt_Depth' not in df.columns:
        raise ValueError("Column 'Gt_Depth' not found in the DataFrame")

    # Iterate over the DataFrame
    outlier_indices = []
    for i in range(len(df)):
        # Define the window range
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, len(df))
        window = df['Gt_Depth'].iloc[start:end]

        # Calculate the median of the window
        mean = np.mean(window)
        # median = np.nanmedian(window)

        # # Check if the current value is an outlier
        if abs(df['Gt_Depth'].iloc[i] - mean) > threshold:
            outlier_indices.append(i)

        # Check if the current value is an outlier
        # if abs(df['Gt_Depth'].iloc[i] - median) > threshold:
        #     outlier_indices.append(i)

    # Check if the outlier indices are in the DataFrame index
    outlier_indices = [idx for idx in outlier_indices if idx in df.index]
    # Now drop the outliers safely
    df_cleaned = df.drop(outlier_indices)
    print(f"Removed {len(outlier_indices)} outlier from data set.")

    return df_cleaned


# def remove_outliers_in_labels_fast(df_chunk, window_size, threshold, target_column_name):
#     # Check if the target column exists
#     if target_column_name not in df_chunk.columns:
#         raise ValueError(f"Column '{target_column_name}' not found in the DataFrame")
#
#     # Initialize the list to collect global outlier indices
#     outlier_indices = []
#
#     for i in range(len(df_chunk)):
#         # Define the window range
#         start = max(i - window_size // 2, 0)
#         end = min(i + window_size // 2 + 1, len(df_chunk))
#         window = df_chunk[target_column_name].iloc[start:end]
#
#         # Calculate the mean of the window
#         mean = np.mean(window)
#         # mean = np.median(window)
#
#         # Check if the current value is an outlier
#         if abs(df_chunk[target_column_name].iloc[i] - mean) > threshold:
#             outlier_indices.append(df_chunk.index[i])  # Collect global index
#
#     # Return the chunk and the outlier indices
#     return df_chunk.index.difference(outlier_indices)
#
#
# def remove_outliers_in_labels(df, window_size, threshold, target_column_name):
#     # Parallelize the operation by splitting the DataFrame into chunks
#     outlier_lists = Parallel(n_jobs=-1)(delayed(remove_outliers_in_labels_fast)(
#         chunk, window_size, threshold, target_column_name)
#                                         for chunk in np.array_split(df, 30)
#                                         )
#
#     # Combine all the outlier indices from each chunk
#     all_outlier_indices = df.index.difference(pd.Index(np.concatenate(outlier_lists)))
#
#     # Drop all outliers from the original DataFrame
#     df_cleaned = df.loc[all_outlier_indices]
#
#     return df_cleaned

def clean_data(df, target_column_name='Gt_Depth', multiplication=1):
    """
    Perform basic cleaning of the dataframe, including outlier removal and binning.

    :param df: The raw dataframe.
    :return: Cleaned dataframe.
    """
    # print("GT ", target_column_name)
    # df = df.dropna(how='all').replace([np.inf, -np.inf], np.nan).dropna().copy()
    # df = df[(df['Gt_Depth'] > minDepth) & (df['Gt_Depth'] <= maxDepth)]
    # df[target_column_name] = df['Gt_Depth'].multiply(100).reset_index(drop=True)

    # Remove rows where all elements are NaN
    df = df.dropna(how='all')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()

    # df2 = df[df['Gt_Depth'] > 0.1]
    df2 = df[df['Gt_Depth'] > 0.35]
    df2 = df2[df2['Gt_Depth'] <= 3]

    df2["Gt_Depth"] = df2["Gt_Depth"].multiply(100)

    df2 = df2.reset_index(drop=True)

    return df2


def split_data_by_subjects(scaled_data, train_size=0.9):
    # Get unique subject IDs
    unique_subjects = scaled_data['SubjectID'].unique()

    # Split subject IDs into training and validation sets
    train_subjects, validation_subjects = train_test_split(unique_subjects, train_size=train_size, shuffle=True)

    # Filter data by subject IDs for training and validation sets
    train_data = scaled_data[scaled_data['SubjectID'].isin(train_subjects)]
    validation_data = scaled_data[scaled_data['SubjectID'].isin(validation_subjects)]

    # now drop the columns as we do not need them anymore
    # Drop the 'SubjectID' column from both datasets
    # train_data = train_data.drop(columns=['SubjectID'])
    # validation_data = validation_data.drop(columns=['SubjectID'])

    return train_data, validation_data


def getIPD(row):
    # Ensure all components (X, Y, Z) for both right and left gaze origins are present
    if 'World_Gaze_Origin_R_Y' not in row:
        posR = [row['World_Gaze_Origin_R_X'], 0.0, row['World_Gaze_Origin_R_Z']]
    else:
        posR = [row['World_Gaze_Origin_R_X'], row['World_Gaze_Origin_R_Y'], row['World_Gaze_Origin_R_Z']]

    if 'World_Gaze_Origin_L_Y' not in row:
        posL = [row['World_Gaze_Origin_L_X'], 0.0, row['World_Gaze_Origin_L_Z']]
    else:
        posL = [row['World_Gaze_Origin_L_X'], row['World_Gaze_Origin_L_Y'], row['World_Gaze_Origin_L_Z']]

    # Check if the positions are exactly the same (which shouldn't happen)
    if posR == posL:
        # print("Warning: Left and right eye positions are identical.")
        return 0.0

    # Calculate the Euclidean distance between the two eye origins (IPD)
    deltaX = posR[0] - posL[0]
    deltaY = posR[1] - posL[1]
    deltaZ = posR[2] - posL[2]

    ipd = math.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    return ipd


def getAngle(row):
    # Extract gaze direction vectors for right and left eyes
    vecR = [row['World_Gaze_Direction_R_X'], row['World_Gaze_Direction_R_Y'], row['World_Gaze_Direction_R_Z']]
    vecL = [row['World_Gaze_Direction_L_X'], row['World_Gaze_Direction_L_Y'], row['World_Gaze_Direction_L_Z']]

    # Compute vector norms (lengths)
    vecR_n = np.linalg.norm(vecR)
    vecL_n = np.linalg.norm(vecL)

    # Check if any vector has zero or near-zero length (i.e., invalid data)
    if vecR_n < 1e-8 or vecL_n < 1e-8:
        print(f"Warning: Vector norm too small (R: {vecR_n}, L: {vecL_n}).")
        return np.nan

    # Compute the dot product
    dot_product = np.dot(vecR, vecL)

    # Normalize dot product to avoid floating point precision issues
    cos_angle = dot_product / (vecR_n * vecL_n)

    # Ensure the cosine value is within the valid range [-1, 1]
    # cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle in radians and convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    # Print the angle for debugging
    # print(f"Angle: {angle_degrees} degrees")

    return angle_degrees


# def getAngle(row):
#     vecR = [row['World_Gaze_Direction_R_X'], row['World_Gaze_Direction_R_Y'], row['World_Gaze_Direction_R_Z']]
#     vecL = [row['World_Gaze_Direction_L_X'], row['World_Gaze_Direction_L_Y'], row['World_Gaze_Direction_L_Z']]
#     vecR_n = np.linalg.norm(vecR)
#     vecL_n = np.linalg.norm(vecL)
#     angle = np.arccos(np.dot(vecR, vecL) / (vecR_n * vecL_n))
#     print("Angle ", angle)
#     return np.degrees(angle)


def global_normalization(data):
    features = data.drop(columns=['SubjectID', 'Gt_Depth'])
    scaler = RobustScaler()  # Global scaler

    normalized_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    data_normalized['SubjectID'] = data['SubjectID'].values
    data_normalized['Gt_Depth'] = data['Gt_Depth'].values
    return data_normalized


def getAngle_GIW(row):
    vecR = [row['World_Gaze_Direction_R_X'], row['World_Gaze_Direction_R_Y'], row['World_Gaze_Direction_R_Z']]
    vecL = [row['World_Gaze_Direction_L_X'], row['World_Gaze_Direction_L_Y'], row['World_Gaze_Direction_L_Z']]
    vecR_n = np.linalg.norm(vecR)
    vecL_n = np.linalg.norm(vecL)
    angle = np.arccos(np.dot(vecR, vecL) / (vecR_n * vecL_n))
    return np.degrees(angle)


def calculate_ipd(row):
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

    return ipd


def getEyeVergenceAngle_GIW(row, ipd):
    """
    Calculate the vergence depth based on the vergence angle and the calculated IPD.

    Args:
        row: A pandas Series row containing the vergence angle.
        ipd: The calculated IPD (from ground truth samples).

    Returns:
        A tuple of (vergence angle, calculated vergence depth).
    """
    vergenceAngle = row['Vergence_Angle']

    # Ensure the vergence angle is numeric
    vergenceAngle = pd.to_numeric(vergenceAngle, errors='coerce')

    if pd.isna(vergenceAngle):
        return np.nan, np.nan  # Invalid vergence angle

    try:
        # Calculate vergence depth using the IPD and vergence angle
        if math.radians(vergenceAngle) != 0:
            depth = ipd / (2 * math.tan(math.radians(vergenceAngle) / 2))
        else:
            depth = np.nan  # Assign NaN for 0 vergence angles
    except ValueError as e:
        print(f"Error in calculating vergence depth: {e}")
        depth = np.nan  # Return NaN in case of calculation errors

    return vergenceAngle, depth


def getEyeVergenceAngle_GIW_old_assumedIPD(row, assumed_ipd=6.3):
    """
    Calculate the vergence depth based on the vergence angle and the assumed IPD.

    Args:
        row: A pandas Series row containing the vergence angle.
        assumed_ipd: The assumed Inter-Pupillary Distance (default 6.3 cm).

    Returns:
        A tuple of (vergence angle, calculated depth) based on the vergence angle and IPD.
    """
    # Extract vergence angle from the row and try to convert it to a float
    vergenceAngle = row['Vergence_Angle']

    # Ensure the vergence angle is numeric
    vergenceAngle = pd.to_numeric(vergenceAngle, errors='coerce')

    if pd.isna(vergenceAngle):
        print(f"Invalid vergence angle: {vergenceAngle}")
        return np.nan, np.nan  # Return NaN for both vergence angle and depth

    # Calculate the vergence depth using the vergence angle
    try:
        if math.radians(vergenceAngle) != 0:
            depth = assumed_ipd / (2 * math.tan(math.radians(vergenceAngle) / 2))
        else:
            depth = np.nan  # Assign NaN for 0 vergence angles
    except ValueError as e:
        print(f"Error in calculating vergence depth: {e}")
        depth = np.nan  # Assign NaN if there's an error in the calculation

    return vergenceAngle, depth


def getEyeVergenceAngle(row):
    vergenceAngle = getAngle(row)
    if vergenceAngle != 0:
        ipd = getIPD(row)
        # Added a check to prevent division by zero if vergenceAngle is extremely small
        depth = ipd / (2 * math.tan(math.radians(vergenceAngle) / 2)) if math.radians(vergenceAngle) != 0 else 0
    else:
        depth = 0.0
    # Convert depth to millimeters by multiplying by 1000 (from meters to mm)
    depth_fin = abs(depth) * 100  # Convert from meters to millimeters
    # print("Depth fin ", depth_fin)
    return vergenceAngle, depth_fin


def subject_wise_normalization(data, unique_subjects, scaler):
    normalized_data_list = []
    for subject in unique_subjects:
        subject_data = data[data['SubjectID'] == subject]
        subject_data_normalized = normalize_subject_data(subject_data, scaler)
        normalized_data_list.append(subject_data_normalized)
    return pd.concat(normalized_data_list, ignore_index=True)


def normalize_subject_data(subject_data, scaler):
    # if subject_data.empty:
    #     print(
    #         f"Subject data is empty for subject ID {subject_data['SubjectID'].iloc[0] if not subject_data.empty else 'Unknown'}")
    #     return pd.DataFrame()  # Return an empty DataFrame to avoid processing

    features = subject_data.drop(columns=['SubjectID', 'Gt_Depth'])
    # print(f"Subject data before normalization (shape: {features.shape}):")  # , features.head())
    normalized_features = scaler.fit_transform(features)
    subject_data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    subject_data_normalized['SubjectID'] = subject_data['SubjectID'].values
    subject_data_normalized['Gt_Depth'] = subject_data['Gt_Depth'].values
    return subject_data_normalized


#
# def subjective_normalization(train_data, validation_data):
#     # Apply global normalization first
#     train_data = global_normalization(train_data)
#     validation_data = global_normalization(validation_data)
#
#     # Then proceed with the existing subject-wise normalization
#     unique_train_subjects = train_data['SubjectID'].unique()
#     unique_validation_subjects = validation_data['SubjectID'].unique()
#
#     # Choose your scaler for subject-wise normalization
#     subject_scaler = RobustScaler()
#
#     # Apply subject-wise normalization
#     training_set_normalized = subject_wise_normalization(train_data, unique_train_subjects, subject_scaler)
#     validation_set_normalized = subject_wise_normalization(validation_data, unique_validation_subjects,
#                                                            subject_scaler)
#
#     return training_set_normalized, validation_set_normalized
#

def createFeatures(data_in, isGIW=False):
    if isGIW:
        # Calculate vergence depth for rows where depth is unknown
        data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(
            *data_in.apply(lambda row: getEyeVergenceAngle_GIW(row, row['IPD']), axis=1))
    else:
        data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(*data_in.apply(getEyeVergenceAngle, axis=1))

    # 1. Depth Normalization
    max_depth = data_in['Vergence_Depth'].max()
    min_depth = data_in['Vergence_Depth'].min()

    data_in['Normalized_Depth'] = (data_in['Vergence_Depth'] - min_depth) / (max_depth - min_depth)

    # 2. Directional Magnitude for Right and Left
    data_in['Directional_Magnitude_R'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values,
        axis=1)
    data_in['Directional_Magnitude_L'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values,
        axis=1)

    # 3. Gaze Direction Cosine Angles (already computed as Vergence_Angle)
    data_in['Cosine_Angles'] = np.cos(np.radians(data_in['Vergence_Angle']))

    # 4. Gaze Point Distance
    data_in['Gaze_Point_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2)

    # 5. Normalized Vergence Angle
    max_angle = data_in['Vergence_Angle'].max()
    min_angle = data_in['Vergence_Angle'].min()
    data_in['Normalized_Vergence_Angle'] = 2 * (
            (data_in['Vergence_Angle'] - min_angle) / (max_angle - min_angle)) - 1

    # 6. Difference in World Gaze Direction
    data_in['Delta_Gaze_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']
    data_in['Delta_Gaze_Y'] = data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']
    data_in['Delta_Gaze_Z'] = data_in['World_Gaze_Direction_R_Z'] - data_in['World_Gaze_Direction_L_Z']

    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    # 1. Angle between Gaze Vectors
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))

    gaze_r = data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values
    gaze_l = data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values
    data_in['Gaze_Vector_Angle'] = [angle_between_vectors(gaze_r[i], gaze_l[i]) for i in range(len(gaze_r))]

    # 2. Gaze Point Depth
    data_in['Gaze_Point_Depth_Difference'] = data_in['World_Gaze_Direction_R_Z'] - data_in[
        'World_Gaze_Direction_L_Z']

    # 3. Relative Changes
    # data_in['Relative_Change_Normalized_Depth'] = data_in['Normalized_Depth'].diff().fillna(0)
    data_in['Relative_Change_Vergence_Angle'] = data_in['Vergence_Angle'].diff().fillna(0)

    # 4. Ratios
    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_Delta_Gaze_XY'] = data_in['Delta_Gaze_X'] / data_in['Delta_Gaze_Y']

    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_World_Gaze_Direction_X'] = data_in['World_Gaze_Direction_R_X'] / data_in[
        'World_Gaze_Direction_L_X']
    data_in['Ratio_World_Gaze_Direction_Y'] = data_in['World_Gaze_Direction_R_Y'] / data_in[
        'World_Gaze_Direction_L_Y']
    data_in['Ratio_World_Gaze_Direction_Z'] = data_in['World_Gaze_Direction_R_Z'] / data_in[
        'World_Gaze_Direction_L_Z']

    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    data_in['Velocity_Gaze_Direction_R_X'] = data_in['World_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Y'] = data_in['World_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Z'] = data_in['World_Gaze_Direction_R_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_X'] = data_in['World_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Y'] = data_in['World_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Z'] = data_in['World_Gaze_Direction_L_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    data_in['Angular_Difference_Gaze_Directions'] = data_in['Gaze_Vector_Angle']  # Already computed above

    data_in['Interaction_Normalized_Depth_Vergence_Angle'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Assuming a rolling window of size 5
    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    data_in['Lag_1_Normalized_Depth'] = data_in['Normalized_Depth'].shift(1).fillna(0)

    # Relative Changes
    data_in['Diff_Normalized_Depth'] = data_in['Normalized_Depth'].diff()

    # Ratios
    data_in['Directional_Magnitude_Ratio'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_X_Ratio'] = data_in['World_Gaze_Direction_R_X'] / data_in['World_Gaze_Direction_L_X']
    data_in['Gaze_Direction_Y_Ratio'] = data_in['World_Gaze_Direction_R_Y'] / data_in['World_Gaze_Direction_L_Y']
    data_in['Gaze_Direction_Z_Ratio'] = data_in['World_Gaze_Direction_R_Z'] / data_in['World_Gaze_Direction_L_Z']

    # Angular Differences
    data_in['Angular_Difference_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']

    # Higher Order Interactions (example)
    data_in['Depth_Angle_Interaction'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Distance between Gaze Points
    data_in['Gaze_Point_Euclidean_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2
    )

    # Angle between Gaze Directions (using dot product)
    dot_product = (
            data_in['World_Gaze_Direction_R_X'] * data_in['World_Gaze_Direction_L_X'] +
            data_in['World_Gaze_Direction_R_Y'] * data_in['World_Gaze_Direction_L_Y'] +
            data_in['World_Gaze_Direction_R_Z'] * data_in['World_Gaze_Direction_L_Z']
    )
    magnitude_R = data_in['Directional_Magnitude_R']
    magnitude_L = data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_Angle'] = np.arccos(dot_product / (magnitude_R * magnitude_L))

    # Calculate acceleration
    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    # print("NaN count per column before dropping NaNs:")
    # print(data_in.isna().sum())
    data_in = data_in.dropna()

    data_in = data_in.replace([np.inf, -np.inf], np.nan)
    data_in = data_in.dropna()
    # Define excluded features
    # Columns to remove
    # excluded_features = [
    #     'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Y', 'World_Gaze_Origin_R_Z',
    #     'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Y', 'World_Gaze_Origin_L_Z',
    #     'IPD'
    # ]
    # data_in = data_in.drop(columns=[col for col in excluded_features if col in data_in.columns], errors='ignore')

    data_in = data_in[input_features]
    # print("Preprocessor: Size of created features: ", data_in.shape)
    # print("Preprocessor: Features ", data_in.columns)
    # Handle NaN and Inf values
    # numerical_columns = data_in.select_dtypes(include=[np.number])
    # data_in[numerical_columns.columns] = numerical_columns.fillna(0)
    # data_in[numerical_columns.columns] = numerical_columns.replace([np.inf, -np.inf], 0)


    return data_in


def createFeatures_new(data_in):
    data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(*data_in.apply(getEyeVergenceAngle, axis=1))

    # 1. Depth Normalization
    max_depth = data_in['Vergence_Depth'].max()
    min_depth = data_in['Vergence_Depth'].min()

    data_in['Normalized_Depth'] = (data_in['Vergence_Depth'] - min_depth) / (max_depth - min_depth)

    # 2. Directional Magnitude for Right and Left
    data_in['Directional_Magnitude_R'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values,
        axis=1)
    data_in['Directional_Magnitude_L'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values,
        axis=1)

    # 3. Gaze Direction Cosine Angles (already computed as Vergence_Angle)
    data_in['Cosine_Angles'] = np.cos(np.radians(data_in['Vergence_Angle']))

    # 4. Gaze Point Distance
    data_in['Gaze_Point_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2)

    # 5. Normalized Vergence Angle
    max_angle = data_in['Vergence_Angle'].max()
    min_angle = data_in['Vergence_Angle'].min()
    data_in['Normalized_Vergence_Angle'] = 2 * (
            (data_in['Vergence_Angle'] - min_angle) / (max_angle - min_angle)) - 1

    # 6. Difference in World Gaze Direction
    data_in['Delta_Gaze_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']
    data_in['Delta_Gaze_Y'] = data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']
    data_in['Delta_Gaze_Z'] = data_in['World_Gaze_Direction_R_Z'] - data_in['World_Gaze_Direction_L_Z']

    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    # 1. Angle between Gaze Vectors
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))

    gaze_r = data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values
    gaze_l = data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values
    data_in['Gaze_Vector_Angle'] = [angle_between_vectors(gaze_r[i], gaze_l[i]) for i in range(len(gaze_r))]

    # 2. Gaze Point Depth
    data_in['Gaze_Point_Depth_Difference'] = data_in['World_Gaze_Direction_R_Z'] - data_in[
        'World_Gaze_Direction_L_Z']

    # 3. Relative Changes
    # data_in['Relative_Change_Normalized_Depth'] = data_in['Normalized_Depth'].diff().fillna(0)
    data_in['Relative_Change_Vergence_Angle'] = data_in['Vergence_Angle'].diff().fillna(0)

    # 4. Ratios
    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_Delta_Gaze_XY'] = data_in['Delta_Gaze_X'] / data_in['Delta_Gaze_Y']

    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_World_Gaze_Direction_X'] = data_in['World_Gaze_Direction_R_X'] / data_in[
        'World_Gaze_Direction_L_X']
    data_in['Ratio_World_Gaze_Direction_Y'] = data_in['World_Gaze_Direction_R_Y'] / data_in[
        'World_Gaze_Direction_L_Y']
    data_in['Ratio_World_Gaze_Direction_Z'] = data_in['World_Gaze_Direction_R_Z'] / data_in[
        'World_Gaze_Direction_L_Z']

    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    data_in['Velocity_Gaze_Direction_R_X'] = data_in['World_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Y'] = data_in['World_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Z'] = data_in['World_Gaze_Direction_R_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_X'] = data_in['World_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Y'] = data_in['World_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Z'] = data_in['World_Gaze_Direction_L_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    data_in['Angular_Difference_Gaze_Directions'] = data_in['Gaze_Vector_Angle']  # Already computed above

    data_in['Interaction_Normalized_Depth_Vergence_Angle'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Assuming a rolling window of size 5
    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    data_in['Lag_1_Normalized_Depth'] = data_in['Normalized_Depth'].shift(1).fillna(0)

    # Relative Changes
    data_in['Diff_Normalized_Depth'] = data_in['Normalized_Depth'].diff()

    # Ratios
    data_in['Directional_Magnitude_Ratio'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_X_Ratio'] = data_in['World_Gaze_Direction_R_X'] / data_in['World_Gaze_Direction_L_X']
    data_in['Gaze_Direction_Y_Ratio'] = data_in['World_Gaze_Direction_R_Y'] / data_in['World_Gaze_Direction_L_Y']
    data_in['Gaze_Direction_Z_Ratio'] = data_in['World_Gaze_Direction_R_Z'] / data_in['World_Gaze_Direction_L_Z']

    # Angular Differences
    data_in['Angular_Difference_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']

    # Higher Order Interactions (example)
    data_in['Depth_Angle_Interaction'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Distance between Gaze Points
    data_in['Gaze_Point_Euclidean_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2
    )

    # Angle between Gaze Directions (using dot product)
    dot_product = (
            data_in['World_Gaze_Direction_R_X'] * data_in['World_Gaze_Direction_L_X'] +
            data_in['World_Gaze_Direction_R_Y'] * data_in['World_Gaze_Direction_L_Y'] +
            data_in['World_Gaze_Direction_R_Z'] * data_in['World_Gaze_Direction_L_Z']
    )
    magnitude_R = data_in['Directional_Magnitude_R']
    magnitude_L = data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_Angle'] = np.arccos(dot_product / (magnitude_R * magnitude_L))

    # Calculate acceleration
    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    # Drop NaN values created by diff() function
    data_in = data_in.dropna()

    data_in = data_in.replace([np.inf, -np.inf], np.nan)
    data_in = data_in.dropna()
    # Define excluded features
    excluded_features = ['World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z', 'World_Gaze_Origin_L_X',
                         'World_Gaze_Origin_L_Z']

    # Remove excluded features
    data_in = data_in.drop(columns=excluded_features)

    data_in = data_in[input_features]
    # print("Preprocessor: Size of created features: ", data_in.shape)
    # print("Preprocessor: Features ", data_in.columns)

    return data_in


def separate_features_and_targets(sequences):
    features = [seq[0] for seq in sequences]
    targets = [seq[1] for seq in sequences]
    return features, targets


def augmentTrainingData(training_set=None):
    training_set = training_set.replace([np.inf, -np.inf], np.nan)
    training_set = training_set.dropna()

    # Separate the columns to be excluded from augmentation
    y_train = training_set['Gt_Depth']
    subject_id = training_set['SubjectID']

    # Drop the columns 'Gt_Depth' and 'SubjectID' from the training set before augmentation
    X_train = training_set.drop(['Gt_Depth', 'SubjectID'], axis=1)
    print("Regression on ", X_train.columns)
    augmentData = True
    if augmentData:
        # ============== DATA AUGMENTATION =============
        X_train_augmented = X_train.copy()  # Ensure a separate copy for augmentation
        X_train_augmented = add_gaussian_noise(X_train_augmented)
        # If you decide to use jitter in the future, uncomment the following line
        # X_train_augmented = jitter(X_train_augmented)

        # The target values remain the same for augmented data
        y_train_augmented = y_train.copy()

        # Use only augmented data
        X_train_combined = X_train_augmented
        y_train_combined = y_train_augmented

    else:
        # Use the original data as the combined dataset
        X_train_combined = X_train
        y_train_combined = y_train

    # Since you're using augmented data only when augmentData is True, or original otherwise,
    # the subject_id_combined handling can stay outside the if-else block as it applies in both cases
    subject_id_combined = subject_id

    # Convert the combined data back to a DataFrame
    training_set_combined_df = pd.DataFrame(X_train_combined, columns=X_train.columns)
    training_set_combined_df['Gt_Depth'] = y_train_combined
    training_set_combined_df['SubjectID'] = subject_id_combined
    training_set_combined_df.dropna(inplace=True)

    return training_set_combined_df


def augment_data(df, noise_level=0.1, shift_max=5, scaling_factor_range=(0.9, 1.1)):
    augmented_data = df.copy()

    # Noise Injection
    for col in df.columns:
        if col != 'SubjectID' and col != 'Gt_Depth':
            noise = np.random.normal(0, noise_level, size=df[col].shape)
            augmented_data[col] += noise

    # Time Shifting
    shift = random.randint(-shift_max, shift_max)
    augmented_data = augmented_data.shift(shift).fillna(method='bfill')

    # Scaling
    scaling_factor = random.uniform(*scaling_factor_range)
    for col in df.columns:
        if col != 'SubjectID' and col != 'Gt_Depth':
            augmented_data[col] *= scaling_factor

    # Example of checking for NaN values before processing
    if augmented_data.isna().any().any():
        # Handle NaN values
        augmented_data = augmented_data.fillna(method='ffill')  # Forward fill as an example

    return augmented_data


def binData(df, isGIW=False):
    # Step 1: Bin the target variable
    num_bins = 60  # You can adjust this number
    df['Gt_Depth_bin'] = pd.cut(df['Gt_Depth'], bins=num_bins, labels=False)

    # Step 2: Calculate mean count per bin
    bin_counts = df['Gt_Depth_bin'].value_counts()
    mean_count = bin_counts.mean()

    # Step 3: Resample each bin
    resampled_data = []
    for bin in range(num_bins):
        bin_data = df[df['Gt_Depth_bin'] == bin]
        bin_count = bin_data.shape[0]

        if bin_count == 0:
            continue  # Skip empty bins

        if bin_count < mean_count:
            # Oversample if count is less than mean
            bin_data_resampled = resample(bin_data, replace=True, n_samples=int(mean_count), random_state=123)
        elif bin_count > mean_count:
            # Undersample if count is more than mean
            bin_data_resampled = resample(bin_data, replace=False, n_samples=int(mean_count), random_state=123)
        else:
            # Keep the bin as is if count is equal to mean
            bin_data_resampled = bin_data

        resampled_data.append(bin_data_resampled)

    # Step 4: Combine back into a single DataFrame
    balanced_df = pd.concat(resampled_data)

    if isGIW:
        balanced_df = sample_from_bins(balanced_df)
        print(balanced_df['Gt_Depth_bin'].value_counts(normalize=True))

    # Optionally, drop the 'Gt_Depth_bin' column if no longer needed
    balanced_df.drop('Gt_Depth_bin', axis=1, inplace=True)
    return balanced_df


def sample_from_bins(df, fraction=0.12):
    # Ensure that each bin has enough data points for sampling
    sampled_data = df.groupby('Gt_Depth_bin').apply(
        lambda x: x.sample(frac=fraction, random_state=110) if len(x) > 1 else x)

    return sampled_data.reset_index(drop=True)
