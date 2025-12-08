from breast_mri_dataset.dataset_utils import DataUtils
from breast_mri_dataset.transform_functions.focal_loss import FocalLoss
from model import MultiModalClassifier
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from monai.visualize import matshow3d
from pathlib import Path


print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(67)
# np.random.seed(67)

class Train:
    def __init__(self):
        # Training logs
        self.training_logs = []
        self.training_accuracy_logs = []
        # Testing Logs
        self.validation_logs = []
        self.testing_logs = []
        self.validation_accuracy_logs = []
        # Cross Validation/Confusion Matrix
        self.pred_T = []
        self.pred_N = []
        self.pred_M = []
        self.true_T = []
        self.true_N = []
        self.true_M = []
        # Hyperparameters
        self.epochs = 20
        self.batch_size = 4
        self.learning_rate = 0.00003
        self.weight_decay = 0.05
        self.image_size = (360, 360)
        self.spacing = (1.0, 1.0, 1.0)
        self.margin = 50
        # Init Training Model
        self.model = MultiModalClassifier(
            fusion_dim=128,
            num_classes=4,
            num_clinical_features=5,
            fusion_strategy='concat'
        ).to(device)
        # Data Utils
        self.data_utils = DataUtils(batch_size=self.batch_size,
                                    image_size=self.image_size,
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
        self.T_loss_fn = FocalLoss(alpha=1, gamma=2)

    def display_batch(self, n_samples=100):
        loader_iter = iter(self.validation_loader)
        displayed = 0

        while displayed < n_samples:
            try:
                batch = next(loader_iter)
            except StopIteration:
                print("Reached end of dataset.")
                break

            patient = batch['Patient ID']
            features = batch["Features"]
            labels = batch["Label"]
            images = batch["Window"]

            batch_size = images.shape[0]
            for i in range(batch_size):
                if displayed >= n_samples:
                    break

                print(f"\nSample {displayed}:")
                print("Label:", labels[i])
                print("Features:", features[i])
                print("Image Path / Folder:", images[i])
                print('Images Shape:', images[i].shape)
                print(f'Batch Shape: {images.shape}')

                # Convert to numpy for visualization
                image_np = images[i].cpu().numpy()  # [D, H, W]
                print('Images Shape:', image_np.shape)
                # Visualize image
                matshow3d(
                    volume=image_np,
                    title=f"T-Stage: {labels[i][0]}, Patient: {patient}",
                    every_n=1,
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
                X_images = batch['Window'].to(device, non_blocking=True)
                X_features = batch['Features'].to(device, non_blocking=True)
                y_labels = batch['Label'].to(device, non_blocking=True)
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                # Decrement T
                label_T = torch.sub(label_T, 1)
                # (T N M) Prediction
                prediction_T = self.model(X_images, X_features)
                # (T N M) loss values
                loss_T = self.T_loss_fn(prediction_T, label_T)
                loss_T.backward()
                # Update Learnable Parameters
                self.optimizer.step()
                # Update loss values for each label
                epoch_loss += loss_T.item()
                # Track training accuracy
                correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                total_correct += correct
                batch_size = y_labels.shape[0]
                predicted_total += batch_size

            # Validation Looping
            val_correct = 0
            val_epoch_loss = 0
            val_predicted = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.validation_loader:
                    # Extract Features from MONAI transforms 'key'
                    X_images = batch['Window'].to(device, non_blocking=True)
                    X_features = batch['Features'].to(device, non_blocking=True)
                    y_labels = batch['Label'].to(device, non_blocking=True)
                    label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                    # Decrement T
                    label_T = torch.sub(label_T, 1)
                    prediction_T = self.model(X_images, X_features)
                    correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                    # (T N M) loss values
                    loss_T = self.T_loss_fn(prediction_T, label_T)
                    total_loss = loss_T
                    val_correct += correct
                    batch_size = self.batch_size
                    val_epoch_loss += total_loss.item()
                    val_predicted += batch_size

            # Log Validation Loss
            validation_accuracy = val_correct / val_predicted
            val_loss = val_epoch_loss / len(self.validation_loader)
            # Log Training Loss
            train_accuracy = total_correct / predicted_total
            train_loss = epoch_loss  / len(self.training_loader)
            self.validation_logs.append(val_loss)
            self.training_logs.append(train_loss)
            self.validation_accuracy_logs.append(validation_accuracy)
            self.training_accuracy_logs.append(train_accuracy)
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
                X_images = batch['Window'].to(device, non_blocking=True)
                X_features = batch['Features'].to(device, non_blocking=True)
                y_labels = batch['Label'].to(device, non_blocking=True)
                # (T N M) Labels
                label_T, label_N, label_M = y_labels[:, 0], y_labels[:, 1], y_labels[:, 2]
                # Decrement T
                label_T = torch.sub(label_T, 1)
                print(label_T)
                prediction_T = self.model(X_images, X_features)
                # Summation of correct predictions across all labels
                correct = (prediction_T.argmax(dim=1) == label_T).sum().item()
                total_correct += correct
                batch_size = y_labels.shape[0] # * 3
                total_predicted += batch_size
                # Predict Report
                self.pred_T.extend(prediction_T.argmax(dim=1).cpu().numpy())
                self.true_T.extend(label_T.cpu().numpy())

        test_accuracy = total_correct / total_predicted
        return test_accuracy

    def results(self):
        min_train_loss = min(self.training_logs)
        max_train_accuracy = max(self.training_accuracy_logs)
        min_val_loss = min(self.validation_logs)
        max_val_accuracy = max(self.validation_accuracy_logs)
        print(f'Lowest Validation Loss: {min_val_loss}'
              f', Epoch: {self.validation_logs.index(min_val_loss)}')
        print(f'Highest Validation Accuracy: {max_val_accuracy}'
              f', Epoch: {self.validation_accuracy_logs.index(max_val_accuracy)}')
        print(f'Lowest Training Loss: {min_train_loss}'
              f', Epoch: {self.training_logs.index(min_train_loss)}')
        print(f'Highest Training Accuracy: {max_train_accuracy}'
              f', Epoch: {self.training_accuracy_logs.index(max_train_accuracy)}')
        # Plot Training Loss
        plt.figure(figsize=(10, 10))
        plt.plot(self.training_logs, c='b', label='Training Loss')
        plt.plot(self.validation_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()
        # Test Accuracy
        test_accuracy = self.test()
        print(f'Test Accuracy: {test_accuracy}')
        cm = classification_report(self.true_T, self.pred_T)
        print(cm)
        # Confusion Matrices
        matrix_T = confusion_matrix(self.true_T, self.pred_T)
        # Confusion Matrix Heatmaps
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix_T, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Tumor Confusion Matrix')
        plt.xlabel('Predicted Tumor')
        plt.ylabel('True Label')
        plt.show()

    def save_model(self):
        torch.save(self.model.state_dict(), '2.5D_cnn.pth')

    def load_model(self):
        # Portable Root
        ROOT = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / 'results' / '2.5D_cnn.pth'
        self.model = MultiModalClassifier(
            fusion_dim=128,
            num_classes=4,
            num_clinical_features=5,
            fusion_strategy='cross_attention'
        ).to(device)
        # Load Model Weights
        self.model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Loading Model from... {MODEL_PATH}')
        self.model.eval()  # Set to evaluation mode


