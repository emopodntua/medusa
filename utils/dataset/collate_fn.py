import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, test3=False):
    num_features = len(batch[0]) - (1 if test3 else 2)

    features = [[] for _ in range(num_features)]
    labels = []
    filenames = []

    # Organize each feature, label, and filename
    for item in batch:
        for i in range(num_features):
            features[i].append(item[i])
        if not test3:
            labels.append(item[-2])
        filenames.append(item[-1])

    collated_features = []

    for feature_list in features:
        first_item = feature_list[0]

        # Pad variable-length tensors (like audio/text)
        if torch.is_tensor(first_item) and first_item.ndim in [1, 2]:
            padded = pad_sequence(feature_list, batch_first=True, padding_value=0)
            collated_features.append(padded)

        # Stack fixed-shape tensors (like embeddings)
        elif torch.is_tensor(first_item):
            stacked = torch.stack(feature_list)
            collated_features.append(stacked)

        # Non-tensors (e.g., strings) passed as is
        else:
            collated_features.append(feature_list)

    # Add labels if present
    if not test3:
        labels = torch.stack(labels).float()
        return collated_features, labels, filenames
    else:
        return collated_features, filenames



def collate_fn_metacls(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch (list of tuples): Each tuple contains (filename, predictions, labels).

    Returns:
        tuple: predictions (tensor), labels (tensor),  filenames (list of str).
    """
    # Unpack the batch
    predictions, labels, filenames = zip(*batch)

    # Convert predictions and labels to tensors
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # print(predictions_tensor, labels_tensor, list(filenames))

    return predictions_tensor, labels_tensor, list(filenames)
