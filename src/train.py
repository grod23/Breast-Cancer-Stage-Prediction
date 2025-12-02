from breast_mri_dataset.dataset_utils import DataUtils
from mm_vit import MultiscaleClassifier
from breast_mri_dataset.focal_loss import FocalLoss
import torch
import torch.nn as nn
# from monai.losses.focal_loss import FocalLoss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from monai.visualize import matshow3d, blend_images
from pathlib import Path
import sys

print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(67)
np.random.seed(67)

class Train:
    def __init__(self):
        # Training logs
        self.training_logs = []
        # Testing Logs
        self.validation_logs = []
        self.testing_logs = []
        # Cross Validation/Confusion Matrix
        self.pred_T = []
        self.pred_N = []
        self.pred_M = []
        self.true_T = []
        self.true_N = []
        self.true_M = []
        # Hyperparameters
        self.epochs = 15
        self.batch_size = 1
        self.learning_rate = 0.00003
        self.weight_decay = 0.3
        self.image_size = (144, 144, 64)
        self.roi_size = (80, 80, 32)
        # self.image_size = (256, 256, 128)
        # self.roi_size = (128, 128, 80)
        self.spacing = (1.0, 1.0, 1.0)
        self.margin = 50
        # Init Training Model
        self.model = MultiscaleClassifier(image_size=self.image_size).to(device)
        # Data Utils
        self.data_utils = DataUtils(batch_size=self.batch_size,
                                    image_size=self.image_size,
                                    roi_size=self.roi_size,
                                    spacing=self.spacing,
                                    margin=self.margin)
        self.training_loader, self.validation_loader, self.testing_loader = (
            self.data_utils.create_dataloaders())
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=3)
        # Class Weights
        self.T_weights, self.N_weights, self.M_weights = self.data_utils.compute_weights()
        # Label specific loss functions
        # self.T_loss_fn = nn.CrossEntropyLoss(weight=self.T_weights.to(device))
        self.T_loss_fn = FocalLoss(alpha=1, gamma=2)

    def display_batch(self, n_samples=5):
        loader_iter = iter(self.training_loader)
        displayed = 0

        while displayed < n_samples:
            try:
                batch = next(loader_iter)
            except StopIteration:
                print("Reached end of dataset.")
                break

            features = batch["Features"]
            labels = batch["Label"]
            images = batch["Folder Path"]
            roi = batch["ROI Crop"]

            batch_size = images.shape[0]
            for i in range(batch_size):
                if displayed >= n_samples:
                    break

                print(f"\nSample {displayed}:")
                print("Label:", labels[i])
                print("Features:", features[i])
                print("Image Path / Folder:", images[i])
                print("ROI shape:", roi[i].shape)

                # Convert to numpy for visualization
                image_np = images[i, 0].cpu().numpy()  # [D, H, W]
                roi_np = roi[i, 0].cpu().numpy()  # [D, H, W]

                # Visualize image
                matshow3d(
                    volume=image_np,
                    title=f"Sample {displayed} - TNM: {labels[i].numpy()}",
                    every_n=5,
                    cmap="gray"
                )
                plt.show()

                # Visualize ROI
                matshow3d(
                    volume=roi_np,
                    title=f"Sample {displayed} - ROI Crop: {roi_np.shape}",
                    every_n=5,
                    cmap="gray"
                )
                plt.show()

                displayed += 1

    def train(self):
        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            self.model.train()
            # Loss tracking for each label
            epoch_loss = 0.0
            total_correct = 0
            predicted_total = 0
            for batch in self.training_loader:
                # Reset Gradients
                self.optimizer.zero_grad()
                # Extract Features from MONAI transforms 'key'
                X_images = batch['Folder Path'].to(device, non_blocking=True)
                X_patches = batch['ROI Crop'].to(device, non_blocking=True)
                X_features = batch['Features'].to(device, non_blocking=True)
                y_labels = batch['Label'].to(device, non_blocking=True)
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                # Decrement T
                label_T = torch.sub(label_T, 1)
                # (T N M) Predictions
                # prediction_T, prediction_N, prediction_M = self.model(X_images, X_features)\
                prediction_T = self.model(X_images, X_patches)
                # print(f'Label: {label_T}')
                # print(f'Prediction: {prediction_T}\n')
                # (T N M) loss values
                loss_T = self.T_loss_fn(prediction_T, label_T)
                # loss_N = self.N_loss_fn(prediction_N, label_N)
                # loss_M = self.M_loss_fn(prediction_M, label_M)
                # print(f'Loss T: {loss_T}')
                # print(f'Loss M: {loss_N}')
                # print(f'Loss M: {loss_M}')
                total_loss = loss_T # + loss_N + loss_M
                # Backpropagation
                total_loss.backward()
                # Update Learnable Parameters
                self.optimizer.step()
                # Update loss values for each label
                epoch_loss += total_loss.item()
                # Track training accuracy
                correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                # correct = (
                #         (prediction_T.argmax(dim=1) == label_T).sum().item() +
                #         (prediction_N.argmax(dim=1) == label_N).sum().item() +
                #         (prediction_M.argmax(dim=1) == label_M).sum().item()
                # )
                total_correct += correct
                batch_size = y_labels.shape[0] #  * 3  # Batch size of 3 but 3 labels per batch so total 9
                predicted_total += batch_size

            # Validation Looping
            val_correct = 0
            val_epoch_loss = 0
            val_predicted = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.validation_loader:
                    # Extract Features from MONAI transforms 'key'
                    X_images = batch['Folder Path'].to(device, non_blocking=True)
                    X_patches = batch['ROI Crop'].to(device, non_blocking=True)
                    X_features = batch['Features'].to(device, non_blocking=True)
                    y_labels = batch['Label'].to(device, non_blocking=True)
                    label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                    # Decrement T
                    label_T = torch.sub(label_T, 1)
                    # prediction_T, prediction_N, prediction_M = self.model(X_images, X_features)
                    prediction_T = self.model(X_images, X_patches)

                    # Summation of correct predictions across all labels
                    correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                    # print(f'Val Prediction: {prediction_T.argmax(dim=1)}')
                    # print(f'Val Ground Truth: {label_T}\n')

                    # correct = (
                    #         (prediction_T.argmax(dim=1) == label_T).sum().item() +
                    #         (prediction_N.argmax(dim=1) == label_N).sum().item() +
                    #         (prediction_M.argmax(dim=1) == label_M).sum().item()
                    # )
                    # print(f'T Prediction: {prediction_T}')
                    # print(f'N Prediction: {prediction_N}')
                    # print(f'M Prediction: {prediction_M}')

                    # (T N M) loss values
                    loss_T = self.T_loss_fn(prediction_T, label_T)
                    # print(f'Loss T: {loss_T}')

                    # loss_N = self.N_loss_fn(prediction_N, label_N)
                    # loss_M = self.M_loss_fn(prediction_M, label_M)
                    # Aggregate losses
                    # if torch.isnan(loss_M):
                    #     total_loss = loss_T + loss_N
                    # else:
                    #     total_loss = loss_T + loss_N + loss_M
                    total_loss = loss_T
                    val_correct += correct
                    batch_size = self.batch_size # * 3  # 3 labels
                    val_epoch_loss += total_loss.item()
                    val_predicted += batch_size

            # Log Validation Loss
            validation_accuracy = val_correct / val_predicted
            val_loss = val_epoch_loss / len(self.validation_loader)
            self.validation_logs.append(val_loss)
            # Log Training Loss
            train_accuracy = total_correct / predicted_total
            train_loss = epoch_loss  / len(self.training_loader)
            self.training_logs.append(train_loss)
            # Outputs
            print(f'Epoch: {epoch}')
            print(f'Training Accuracy: {train_accuracy}')
            print(f'Validation Accuracy: {validation_accuracy}')
            print(f'Train Loss: {train_loss}')
            print(f'Validation Loss: {val_loss}')

            # Update the scheduler
            self.scheduler.step(val_loss)

    def test(self):
        total_correct = 0
        total_predicted = 0

        # Set model to evaluation
        self.model.eval()
        with torch.no_grad():
            for batch in self.testing_loader:
                # Extract Features from MONAI transforms 'key'
                X_images = batch['Folder Path'].to(device, non_blocking=True)
                X_features = batch['Features'].to(device, non_blocking=True)
                y_labels = batch['Label'].to(device, non_blocking=True)
                print(f'Image Shape: {X_images.shape}')
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                # Decrement T
                label_T = torch.sub(label_T, 1)
                # prediction_T, prediction_N, prediction_M = self.model(X_images, X_features)
                prediction_T = self.model(X_images)
                # Summation of correct predictions across all labels
                correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                # correct = (
                #         (prediction_T.argmax(dim=1) == label_T).sum().item() +
                #         (prediction_N.argmax(dim=1) == label_N).sum().item() +
                #         (prediction_M.argmax(dim=1) == label_M).sum().item()
                # )
                total_correct += correct
                batch_size = y_labels.shape[0] # * 3
                total_predicted += batch_size
                print(f'Test Batch - Correct: {correct} / {batch_size}')

                # Predict Report
                self.pred_T.extend(prediction_T.argmax(dim=1).cpu().numpy())
                # self.pred_N.extend(prediction_N.argmax(dim=1).cpu().numpy())
                # self.pred_M.extend(prediction_M.argmax(dim=1).cpu().numpy())
                self.true_T.extend(label_T.cpu().numpy())
                # self.true_N.extend(label_N.cpu().numpy())
                # self.true_M.extend(label_M.cpu().numpy())

        test_accuracy = total_correct / total_predicted
        return test_accuracy

    def results(self):
        # Plot Training Loss
        plt.figure(figsize=(10, 10))
        plt.plot(self.training_logs, c='b', label='Training Loss')
        plt.plot(self.validation_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()

        # Confusion Matrices
        matrix_T = confusion_matrix(self.true_T, self.pred_T)
        # matrix_N = confusion_matrix(self.true_N, self.pred_N)
        # matrix_M = confusion_matrix(self.true_M, self.pred_M)
        # Confusion Matrix Heatmaps

        # Tumor CM
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix_T, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Tumor Confusion Matrix')
        plt.xlabel('Predicted Tumor')
        plt.ylabel('True Label')
        plt.show()
        # Nodal CM
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(matrix_N, annot=True, fmt='d', cmap='Blues', cbar=True)
        # plt.title('Nodal Confusion Matrix')
        # plt.xlabel('Predicted Node')
        # plt.ylabel('True Label')
        # plt.show()
        # Metastasis CM
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(matrix_M, annot=True, fmt='d', cmap='Blues', cbar=True)
        # plt.title('Metastasis Confusion Matrix')
        # plt.xlabel('Predicted Metastasis')
        # plt.ylabel('True Label')
        # plt.show()

        print(f'Lowest Validation Loss: {min(self.validation_logs)}'
              f', Epoch: {self.validation_logs.index(min(self.validation_logs))}')
        print(f'Lowest Training Loss: {min(self.training_logs)}'
              f', Epoch: {self.training_logs.index(min(self.training_logs))}')

        # Test Accuracy
        test_accuracy = self.test()
        print(f'Test Accuracy: {test_accuracy}')

    def save_model(self):
        torch.save(self.model.state_dict(), 'vit_model.pth')

    def load_model(self):
        # Portable Root
        ROOT = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / "results" / "vit_model.pth"
        self.model = MultiscaleClassifier(image_size=self.image_size).to(device)
        # Load Model Weights
        self.model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Loading Model from... {MODEL_PATH}')
        self.model.eval()  # Set to evaluation mode


