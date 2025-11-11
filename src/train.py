from breast_mri_dataset.dataset_utils import DataUtils
from mm_vit import MultiModalTransformer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from monai.visualize import matshow3d, blend_images

import sys

print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train:
    def __init__(self):
        # Training logs
        self.training_logs = []
        self.validation_logs = []
        self.testing_logs = []
        self.model = MultiModalTransformer()
        # Hyperparameters
        self.epochs = 10
        self.batch_size = 1
        self.learning_rate = 0.0001
        self.weight_decay = 0.001
        self.dropout_rate = 0.0
        # Data Utils
        self.data_utils = DataUtils(batch_size=self.batch_size)
        self.training_loader, self.testing_loader = self.data_utils.create_dataloaders()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)

    def display_batch(self):
        # Get one batch
        batch = next(iter(self.training_loader))
        features = batch["features"]  # shape: [batch_size, num_features]
        labels = batch["label"]  # shape: [batch_size, num_labels]
        images = batch["image_paths"]  # or "image_paths", depending on your pipeline

        print("Images shape:", images.shape)  # Should be [B, C, D, H, W]
        print("Labels shape:", labels.shape)  # Should be [B, 3] if TNM
        print("Features shape:", features.shape)  # Should be [B, num_features]
        print("Images dtype:", images.dtype)
        print("Labels dtype:", labels.dtype)
        print("Features dtype:", features.dtype)
        print(labels)
        print(features)
        print(images)

        # Visualize first sample
        image = batch['image_paths'][0, 0].cpu().numpy()  # [D, H, W]
        label = batch['label'][0]

        matshow3d(
            volume=image,
            title=f"TNM: {label.numpy()}",
            every_n=1,  # Show every 6th slice
            cmap='gray'
        )

        plt.show()

    def train(self):
        pass
