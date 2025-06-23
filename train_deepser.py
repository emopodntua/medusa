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

from transformers import AutoModel, RobertaTokenizer, RobertaForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, RobertaForMaskedLM, WhisperFeatureExtractor, WhisperTokenizer
from utils.dataloader_multi import WavLMDataLoaderJoint
from utils.focal_loss import *
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

features = config.get("features")
mlp_outs = config.get("mlp_outs")
pooling_type = config.get("pooling", "mean")

training = config.get("training", {})
batch_size = training.get("batch_size", 32)
accumulation_step = training.get("accumulation_steps", 8)
learning_rate = training.get("learning_rate", 1e-5)
epochs = training.get("epochs", 50)
mixup = training.get("mixup", 0.0)
patience = training.get("patience", 8)
switch_epoch = training.get("switch_epoch", 4)

loss = config.get("loss", {})
alpha = loss.get("alpha", 0.5)
l1 = loss.get("l1", 1.5)
l2 = loss.get("l2", 0.4)

paths = config.get("paths", {})
label_path = paths.get("label_path", "./labels/labels.csv")
label_no_XO_path = paths.get("label_no_XO_path", "./labels/balanced_soft_labels_multi_no_XO.csv")
label_only_XO_path = paths.get("label_only_XO_path", "./labels/balanced_soft_labels_multi_XO.csv")

dataset_type = paths.get("dataset_type", "Final")

model_ckpt_path = paths.get("model_ckpt_path", "./ckpts/model.pt")
model_output_dir = paths.get("model_output_dir", "./ckpts")
os.makedirs(model_output_dir, exist_ok=True)

model_name = paths.get("model_name", "DeepSER")



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

embed_dims = []

if features is None or not isinstance(features, list) or len(features) == 0:
    raise ValueError("Features must be a non-empty list of feature directories.")

for feature_folder in features:
    # Read embedding dimensions by loading a .npy sample from the features folder
    if not os.path.exists(feature_folder):
        raise FileNotFoundError(f"Feature folder {feature_folder} does not exist.")
    # Get the embedding dimensions from the first file in the audio features directory
    feature_files = glob.glob(os.path.join(feature_folder, "*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {feature_folder}.")
    sample_file = feature_files[0]
    sample_feat = np.load(sample_file)
    # Get the embedding dimensions
    embed_dim = sample_feat.shape[-1]  # Assuming the last dimension is the feature size
    embed_dims.append(embed_dim)





train_dataset = utils.DeepSERDataset(label_path, #balanced_soft_labels_multi_new.csv
                            features,
                            dtype=dataset_type)


# Unbalanced dev set 
dev_dataset = utils.DeepSERDataset(label_path,
                            features,
                            dtype='Development')
print(len(dev_dataset))

dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=cur_bs,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn
        )

print(len(dev_dataloader))

train_dataloader_full = DataLoader(
            train_dataset,
            batch_size=cur_bs,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn
        )

model = net.DeepSER(embed_dims, mlp_outs, mixup)

if model_ckpt_path is not None:
    print(f"Loading model from {model_ckpt_path}")
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

model.cuda()
opt = torch.optim.AdamW(model.parameters(), learning_rate)
opt.zero_grad(set_to_none=True)


min_epoch = 0
min_loss = 1e10
reduction = 'mean'
task_type = 'multi-label'
num_classes = 8
centropy = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction=reduction)
mse_loss = torch.nn.MSELoss(reduction='mean')
best_f1 = 0.0  # Track the best F1 score
no_improve_epochs = 0  # Counter for epochs without improvement


# maybe remove that 
lm = utils.LogManager()
lm.alloc_stat_type_list(["train_loss"])
lm.alloc_stat_type_list(["dev_loss"])


for epoch in range(epochs):
    print("Epoch: ", epoch)
    lm.init_stat()
    model.train()
    batch_cnt = 0
    
    dataset_balanced = utils.DeepSERDataset(label_no_XO_path, # balanced_soft_labels_multi_no_XO.csv
                                features,
                                dtype=dataset_type, 
                                balance=True)
    dataset_XO = utils.DeepSERDataset(label_only_XO_path, # balanced_soft_labels_multi_XO.csv
                                    features,
                                    dtype=dataset_type, 
                                    balance=False)
    dataset_XO_new = sample_and_concat_datasets([dataset_XO], num_samples=len(dataset_balanced)//num_classes)
    datasets = ConcatDataset([dataset_balanced, dataset_XO_new])
    
    train_dataloader = DataLoader(
        datasets,
        batch_size=cur_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )


    if epoch < switch_epoch:
        train_dataloader_sel = train_dataloader_full
        centropy = torch.nn.CrossEntropyLoss(weight=class_weights_tensor,reduction='mean')
    else:
        if epoch == switch_epoch:
            print("Switching to balanced training...")
        train_dataloader_sel = train_dataloader
        centropy = torch.nn.CrossEntropyLoss(reduction='mean')

        opt = torch.optim.AdamW(model.parameters(), 1e-06)
        opt.zero_grad(set_to_none=True)


    for batch in tqdm(train_dataloader_sel):
        # print(type(batch))
        # print(len(batch[0]))
        x, y, filenames = batch

        x = [sample.cuda(non_blocking=True).float() for sample in x]
        y = y.cuda(non_blocking=True).float()

        # Handle labels
        if mixup:
            emo_pred_cat, emo_pred_adv, y_mixup = model(x, labels=y, dev=False)
            y_cat = y_mixup[:,:8]
            y_adv = y_mixup[:,8:]
        else:
            y_cat = y[:,:-3]
            y_cat = y_cat.max(dim=1)[1]
            y_cat = y_cat.cuda(non_blocking=True).long()
            y_adv = y[:,-3:]
            y_adv = y_adv.cuda(non_blocking=True).float()
            emo_pred_cat, emo_pred_adv = model(x, dev=False)

    # Calculate loss
    loss = l1*centropy(emo_pred_cat, y_cat) + l2*mse_loss(emo_pred_adv, y_adv)
    total_loss = loss / accumulation_step
    total_loss.backward()
    
    if (batch_cnt+1) % accumulation_step == 0 or (batch_cnt+1) == len(train_dataloader):
        opt.step()
        opt.zero_grad(set_to_none=True)
    
    batch_cnt += 1
    # Logging
    lm.add_torch_stat("train_loss", loss)

    
    model.eval()
    total_pred_cat = []
    total_y_cat = []
    total_pred_adv = []
    total_y_adv = []
    total_utt = []


    print('Unbalanced development set:\n')
    for batch in tqdm(dev_dataloader):
        # print(type(batch))
        # print(len(batch[0]))
        x, y, filenames = batch

        x = [sample.cuda(non_blocking=True).float() for sample in x]
        y = y.cuda(non_blocking=True).float()

        y_cat = y[:,:-3]
        y_cat = y_cat.max(dim=1)[1]
        y_cat = y_cat.cuda(non_blocking=True).long()
        y_adv = y[:,-3:]
        y_adv = y_adv.cuda(non_blocking=True).float()
        emo_pred_cat, emo_pred_adv = model(x, dev=False)

        with torch.no_grad():
            emo_pred_cat, emo_pred_adv = model(x, dev=True)

        total_pred_cat.append(emo_pred_cat)
        total_y_cat.append(y_cat)
        total_pred_adv.append(emo_pred_adv)
        total_y_adv.append(y_adv)
        total_utt.append(filenames)

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

    if f1 > best_f1:
        best_f1 = f1
        no_improve_epochs = 0  # Reset counter for no improvement

        print(f"New best F1: {best_f1:.4f} at epoch {epoch}. Saving model.")
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s).")

    torch.save(
        model.state_dict(),
        os.path.join(model_output_dir, model_name + f'-f1_{f1:.4f}-ep_{epoch}.pt')
    )

    # Early stopping if F1 doesn't improve for `patience` epochs
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
        break
