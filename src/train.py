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
from pathlib import Path

print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(67)
np.random.seed(67)

class Train:
    def __init__(self):
        # Training logs
        self.training_logs = []
        self.validation_logs = []
        self.testing_logs = []
        # Hyperparameters
        self.epochs = 120
        self.batch_size = 3
        self.learning_rate = 0.0003
        self.weight_decay = 0.001
        self.dropout_rate = 0.0
        self.image_size = (256, 256, 160)
        # Patch Learning
        self.roi_size = (128, 128, 128)
        self.spacing = (1.0, 1.0, 1.0)
        # Init Training Model
        self.model = MultiModalTransformer().to(device)
        # Data Utils
        self.data_utils = DataUtils(batch_size=self.batch_size, image_size=self.image_size,
                                    spacing=self.spacing, roi_size=self.roi_size)
        self.training_loader, self.testing_loader = self.data_utils.create_dataloaders()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def display_batch(self):
        # Get one batch
        batch = next(iter(self.training_loader))
        features = batch["features"]  # shape: [batch_size, num_features]
        labels = batch["label"]  # shape: [batch_size, num_labels]
        images = batch["image_paths"]

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
        # Training Loop
        total_correct = 0
        predicted_total = 0
        self.model.train()
        for epoch in range(self.epochs):
            # Loss tracking for each label
            epoch_loss = 0.0
            for batch in self.training_loader:
                # Reset Gradients
                self.optimizer.zero_grad()
                # Extract Features from MONAI transforms 'key'
                X_images = batch['image_paths'].to(device, non_blocking=True)
                X_features = batch['features'].to(device, non_blocking=True)
                y_labels = batch['label'].to(device, non_blocking=True)
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                # (T N M) Predictions
                prediction_T, prediction_N, prediction_M = self.model(X_images, X_features)
                print(f'Prediction Tumor: {prediction_T}')
                print(f'Prediction Node: {prediction_N}')
                print(f'Prediction Metastasis: {prediction_M}')
                # (T N M) loss values
                loss_T = self.loss_fn(prediction_T, label_T)
                loss_N = self.loss_fn(prediction_N, label_N)
                loss_M = self.loss_fn(prediction_M, label_M)
                print(f'Loss Tumor: {loss_T.item()}')
                print(f'Loss Node: {loss_N.item()}')
                print(f'Loss Metastasis: {loss_M.item()}')
                # Aggregate losses
                total_loss = loss_T + loss_N + loss_M
                print(f'Total Loss: {total_loss.item()}')
                # Backpropagation
                total_loss.backward()
                # Update Learnable Parameters
                self.optimizer.step()
                # Update loss values for each label
                epoch_loss += total_loss.item()
                # Track training accuracy
                correct = (
                        (prediction_T.argmax(dim=1) == label_T).sum().item() +
                        (prediction_N.argmax(dim=1) == label_N).sum().item() +
                        (prediction_M.argmax(dim=1) == label_M).sum().item()
                )
                total_correct += correct
                batch_size = y_labels.shape[0] * 3  # Batch size of 3 but 3 labels per batch so total 9
                predicted_total += batch_size

            # Log Training Loss
            train_accuracy = total_correct / predicted_total
            train_loss = epoch_loss  / len(self.training_loader)
            self.training_logs.append(train_loss)
            print(f'Epoch: {epoch}')
            print(f'Training Accuracy: {train_accuracy}')
            print(f'Train Loss: {train_loss}')


    def test(self):
        total_correct = 0
        total_predicted = 0

        # Set model to evaluation
        self.model.eval()
        with torch.no_grad():
            for batch in self.testing_loader:
                # Extract Features from MONAI transforms 'key'
                X_images = batch['image_paths'].to(device, non_blocking=True)
                X_features = batch['features'].to(device, non_blocking=True)
                y_labels = batch['label'].to(device, non_blocking=True)
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                prediction_T, prediction_N, prediction_M = self.model(X_images, X_features)
                # Summation of correct predictions across all labels
                correct = (
                        (prediction_T.argmax(dim=1) == label_T).sum().item() +
                        (prediction_N.argmax(dim=1) == label_N).sum().item() +
                        (prediction_M.argmax(dim=1) == label_M).sum().item()
                )
                total_correct += correct
                batch_size = y_labels.shape[0] * 3
                total_predicted += batch_size
                print(f'Test Batch - Correct: {correct} / {batch_size}')
        test_accuracy = total_correct / total_predicted
        return test_accuracy

    def results(self):
        # Test Accuracy
        test_accuracy = self.test()
        print(f'Test Accuracy: {test_accuracy}')

        # Plot Training Loss
        plt.figure(figsize=(10, 10))
        plt.plot(self.training_logs, label='Training Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()

    def save_model(self):
        torch.save(self.model.state_dict(), 'vit_model.pth')

    def load_model(self):
        # Portable Root
        ROOT = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / "results" / "vit_model.pth"
        self.model = MultiModalTransformer().to(device)
        # Load Model Weights
        self.model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Loading Model from... {MODEL_PATH}')
        self.model.eval()  # Set to evaluation mode
