import numpy as np
import pickle as pk
import torch.utils as torch_utils
from . import normalizer
import torch
import os
from torch.utils.data import Dataset, Subset, ConcatDataset
import pandas as pd
import random
from tqdm import tqdm
from typing import Union
from sklearn.utils import resample

# Utils
def balance_dataset(df):
    """
    Balances the dataset by undersampling majority classes.

    Args:
        df (pd.DataFrame): Input dataframe with labels.

    Returns:
        pd.DataFrame: Balanced dataframe.
    """
    df["Dominant_Emotion"] = df.iloc[:, 1:9].idxmax(axis=1)  # Assuming emotions are in cols 1-8
    class_counts = df["Dominant_Emotion"].value_counts()
    # print(class_counts)
    min_samples = class_counts.min()

    # Undersample majority classes
    balanced_dfs = []
    for emotion in class_counts.index:
        class_df = df[df["Dominant_Emotion"] == emotion]
        if len(class_df) > min_samples:
            class_df = resample(class_df, replace=False, n_samples=min_samples, random_state=42)
        balanced_dfs.append(class_df)

    balanced_df = pd.concat(balanced_dfs)

    return balanced_df.drop(columns=["Dominant_Emotion"])


def sample_and_concat_datasets(datasets, num_samples):
    """
    Samples an equal number of samples from each dataset and combines them into a single dataset.

    Args:
        datasets (list of Dataset): A list of PyTorch datasets.
        num_samples (int): The number of samples to select from each dataset.

    Returns:
        ConcatDataset: A concatenated dataset containing the sampled subsets from each input dataset.
    """
    print("Creating dataset with max size per emotion = ", num_samples)
    sampled_subsets = []

    for dataset in datasets:
        # Ensure num_samples does not exceed the dataset size
        dataset_size = len(dataset)
        if num_samples > dataset_size:
            # raise ValueError(f"Requested num_samples ({num_samples}) exceeds the size of the dataset ({dataset_size}).")
            indices = np.random.choice(dataset_size, dataset_size, replace=False)
        # Randomly sample indices
        else:
            indices = np.random.choice(dataset_size, num_samples, replace=False)
        
        # Create a subset with the sampled indices
        subset = Subset(dataset, indices)
        sampled_subsets.append(subset)
    
    # Combine all subsets into a single dataset using ConcatDataset
    combined_dataset = ConcatDataset(sampled_subsets)
    return combined_dataset


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]




class DeepSERDataset(Dataset):
    def __init__(self, csv_path, dirs, dtype, balance=False):
        """
        Custom dataset for loading N unimodal features with optional balancing.

        Args:
            csv_path (str): Path to CSV file.
            dirs (list): List of directories containing unimodal feature paths (str).
            dtype (str): Dataset type ('Train', 'Development', 'Test', 'Final', 'Full').
            balance (bool): Whether to balance the dataset.
        """
        self.dtype = dtype
        self.dirs = dirs

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Filter based on dtype
        if dtype == 'Full':
            df = df[df['Split_Set'].isin(['Train', 'Development', 'Test'])]
        elif dtype == 'Final':
            df = df[df['Split_Set'].isin(['Train', 'Test'])]
        else:
            df = df[df['Split_Set'] == dtype]

        # Balance the dataset if required
        if balance and dtype in ['Train', 'Development', 'Final', 'Full']:
            df = balance_dataset(df)

        self.file_list = df['FileName'].tolist()
        self.labels = df.iloc[:, 1:-1].values  # All columns except FileName and Split_Set

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        full_file_name = self.file_list[idx]
        file_name = full_file_name[:-4]  # Remove the '.wav' extension

        # Load data
        data = [np.load(os.path.join(dir, f"{file_name}.npy")) for dir in self.dirs]

        # Convert to tensors
        tensors = [torch.tensor(datum, dtype=torch.float32) for datum in data]
  
        if self.dtype == 'Test3':
            return *tensors, full_file_name
        else:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return *tensors, label_tensor, full_file_name




def softmax_first_eight(row):
    # Convert the string of comma-separated values to a list of floats
    values = [float(p) for p in row.split(",")]
    
    # Apply softmax to the first 8 elements
    first_8 = values[:8]
    softmaxed = np.exp(first_8) / np.sum(np.exp(first_8))
    
    # Combine softmaxed elements with the rest of the values
    return softmaxed.tolist() + values[8:]





class MetaclsDataset(Dataset):
    def __init__(self, gt_path, csv_paths, dtype, balance=False):
        """
        Initialize the dataset.

        Args:
            gt_path (str): Path to the ground truth CSV file containing 'FileName' and labels.
            csv_paths (list of str): List of paths to CSV files containing predictions.
            dtype (str): Data split type ('Train', 'Test', 'Development', 'Final').
        """
        self.dtype = dtype

        # Load and filter ground truth file
        if dtype != 'Test3':
            df = pd.read_csv(gt_path)

        if dtype == 'Full':
            df = df[(df['Split_Set'] == 'Train') | (df['Split_Set'] == 'Development') | (df['Split_Set'] == 'Test')]
        elif dtype == 'Final':
            df = df[(df['Split_Set'] == 'Train') | (df['Split_Set'] == 'Test')]
        elif dtype == 'Test3':
            df = pd.DataFrame([f'MSP-PODCAST_test3_{i:04}.wav' for i in range(1,3201)], columns=['FileName'])
        else:
            df = df[df['Split_Set'] == dtype]

        df = df.sort_values(by='FileName')

        # Read predictions from all CSVs
        self.predictions = []
        for csv_path in csv_paths:
            df_preds = pd.read_csv(csv_path)
            df_preds = df_preds.sort_values(by='FileName')
            df_preds = pd.merge(df_preds, df, on='FileName', how='inner')
            # Assuming predictions are in the second column as a string of comma-separated values
            preds = df_preds.iloc[:, 1].apply(lambda x: softmax_first_eight(x)).tolist()
            self.predictions.append(preds)

        # Get the file names and labels (first 8 columns)
        self.file_list = df_preds['FileName'].tolist()
        self.labels = df.iloc[:, 1:12].values.tolist()

        # print(self.file_list[:3], self.labels[:3])

        self.predictions = [
            [p for preds in sample_preds for p in preds]
            for sample_preds in zip(*self.predictions)
        ]

        # print()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        preds = self.predictions[idx]  # List of predictions from all CSVs
        
        if self.dtype == 'Test3':
            return preds, file_name
        else:
            labels = self.labels[idx]  # Labels for the sample
            return preds, labels, file_name
