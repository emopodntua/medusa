import os
import sys
import argparse
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import csv
from time import perf_counter

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.optim as optim
import importlib
import random

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils

from utils.dataset.collate_fn import *
from utils.dataset import *

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YAML config file
with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)

seed = config.get("seed", 42)
utils.set_deterministic(seed)

csv_result_paths = config.get("csv_result_paths")
mlp_outs = config.get("mlp_outs", [])

training = config.get("training", {})
batch_size = training.get("batch_size", 32)
accumulation_step = training.get("accumulation_steps", 8)

loss = config.get("loss", {})
alpha = loss.get("alpha", 0.5)
l1 = loss.get("l1", 1.5)
l2 = loss.get("l2", 0.4)

paths = config.get("paths", {})
label_path = paths.get("label_path", "./labels/labels.csv")
label_no_XO_path = paths.get("label_no_XO_path", "./labels/balanced_soft_labels_multi_no_XO.csv")
label_only_XO_path = paths.get("label_only_XO_path", "./labels/balanced_soft_labels_multi_XO.csv")

dataset_type = config.get("dataset_type")

model_ckpt_path = paths.get("model_ckpt_path")

model_name = paths.get("model_name", "DeepSER")

csv_output_dir = paths.get("csv_output_dir", "./output/results.csv") # CSV output path
os.makedirs(csv_output_dir, exist_ok=True)




df = pd.read_csv(label_path)
# Filter out only 'Train' samples
# train_df = df[df['Split_Set'] == 'Train']
train_df = df[(df['Split_Set'] == 'Train') | (df['Split_Set'] == 'Test')]

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()
# Total number of samples
total_samples = len(train_df)
# Calculate class weights
class_weights = {cls: (total_samples / (len(classes) * freq))**(alpha) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
print(class_weights)
# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]
# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device=device, dtype=torch.float)
# Print or return the tensor
print(class_weights_tensor)


cur_bs = batch_size // accumulation_step


# Dataset and DataLoader
dev_dataset = utils.MetaclsDataset(label_path,
                            csv_result_paths,
                            dtype=dataset_type)
print(len(dev_dataset))

dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=cur_bs,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn_metacls
        )

print(len(dev_dataloader))



num_classes = 8
num_emo = 3
num_total = num_classes + num_emo

input_size = len(csv_result_paths) * num_total # (number of DeepSER models) * (number of outputs per model)

layers = []

if mlp_outs is None or len(mlp_outs) == []:
    layers = nn.Linear(input_size, num_total)
else:
    model_dims = [input_size] + mlp_outs + [num_total]
    for i in range(len(model_dims) - 1):
        layers.append(nn.Linear(model_dims[i], model_dims[i+1]))
        layers.append(nn.ReLU())

model = nn.Sequential(*layers)

if model_ckpt_path is not None:
    print(f"Loading model from {model_ckpt_path}")
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
else:
    raise ValueError("Model checkpoint path is not provided or does not exist.")

model.cuda()

lm = utils.LogManager()
lm.alloc_stat_type_list(["dev_loss"])

min_epoch=0
min_loss=1e10

lm.init_stat()

INFERENCE_TIME=0

reduction = 'mean'
task_type = 'multi-label'
num_classes = 8

centropy = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='mean')
mse_loss = torch.nn.MSELoss(reduction='mean')
    
model.eval()
total_pred_cat = []
total_y_cat = []
total_pred_adv = []
total_y_adv = []
total_utt = []


print('Evaluation on ' + dataset_type + ' set:\n')
for batch in tqdm(dev_dataloader):

    x, y, filenames = batch

    stime = perf_counter()

    x = torch.tensor(x).cuda(non_blocking=True).float()
    y = y.cuda(non_blocking=True).float()

    y_cat = y[:,:-3]
    y_cat = y_cat.max(dim=1)[1]
    y_cat = y_cat.cuda(non_blocking=True).long()
    y_adv = y[:,-3:]
    y_adv = y_adv.cuda(non_blocking=True).float()

    with torch.no_grad():
        emo_pred = model(x)

    emo_pred_cat = emo_pred[:,:8]
    emo_pred_adv = emo_pred[:,8:]

    total_pred_cat.append(emo_pred_cat)
    total_y_cat.append(y_cat)
    total_pred_adv.append(emo_pred_adv)
    total_y_adv.append(y_adv)
    total_utt.append(filenames)

    etime = perf_counter()
    INFERENCE_TIME += (etime-stime)

data = []

# CCC calculation
total_pred_cat2 = torch.cat(total_pred_cat, 0)
total_y_cat = torch.cat(total_y_cat, 0)

pred_classes = torch.argmax(total_pred_cat2, dim=1).cpu().numpy()
true_classes = total_y_cat.cpu().numpy()

report = classification_report(true_classes, pred_classes, digits=4)
print(report)

# Compute F1-score
f1 = f1_score(true_classes, pred_classes, average='macro')  # Use 'weighted' to account for class imbalance
print(f"F1-Score: {f1:.4f}")

loss = l1*centropy(emo_pred_cat, y_cat) + l2*mse_loss(emo_pred_adv, y_adv)

# Logging
lm.add_torch_stat("dev_loss", loss)

# Save model
lm.print_stat()

dev_loss = lm.get_stat("dev_loss")

for preds_cat_batch, preds_adv_batch, utts_batch in zip(total_pred_cat, total_pred_adv, total_utt):
    # Iterate over each item within the batch
    for pred_cat, pred_adv, utt in zip(preds_cat_batch, preds_adv_batch, utts_batch):
        pred_cat_values = ', '.join([f'{val:.4f}' for val in pred_cat.cpu().numpy().flatten()])
        pred_adv_values = ', '.join([f'{val:.4f}' for val in pred_adv.cpu().numpy().flatten()])
        pred_values = pred_cat_values + ', ' + pred_adv_values
        data.append([utt, pred_values])  # Ensure utt is a string


# Writing to CSV file
csv_filename = os.path.join(csv_output_dir, model_name + '_' + dataset_type + '.csv')
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['FileName', 'Prediction'])
    writer.writerows(data)
