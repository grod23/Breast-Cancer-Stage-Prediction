from breast_mri_dataset.dataset_utils import DataUtils
from breast_mri_dataset.skl_dataset import SKL_Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import sys

print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train:
    def __init__(self):
        self.training_logs = []
        self.validation_logs = []
        self.data_utils = DataUtils()
        self.training_loader, self.testing_loader = self.data_utils.create_dataloaders()
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        # Get one batch
        batch = next(iter(self.training_loader))
        images, labels, features = batch  # unpack if your transform returns these keys

        print("Images shape:", images.shape)  # Should be [B, C, D, H, W]
        print("Labels shape:", labels.shape)  # Should be [B, 3] if TNM
        print("Features shape:", features.shape)  # Should be [B, num_features]
        print("Images dtype:", images.dtype)
        print("Labels dtype:", labels.dtype)
        print("Features dtype:", features.dtype)

