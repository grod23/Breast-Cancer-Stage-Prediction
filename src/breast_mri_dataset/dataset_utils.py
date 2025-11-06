from .dataset import Breast_MRI
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import sys

def load_sequences_dict():
    # Load
    sequences = joblib.load("breast_mri_dataset/sequence_data.joblib")
    return sequences

class DataUtils:
    def __init__(self):
        self.sequences = load_sequences_dict()

    def get_train_split(self):
        # X/input
        patient_ids = [patient['patient_id'] for patient in self.sequences]
        print(patient_ids)
        # y/truth label
        labels = [patient['label'] for patient in self.sequences]
        print(len(patient_ids))
        print(len(labels))

        # Train Test Split
        X_train_ids, X_test_ids, y_train, y_test = train_test_split(patient_ids, labels, test_size=0.20,
                                                                    random_state=30)

        print(y_train)
        # Currently list of patient ids
        # Want list of sequences
        # Filter sequences for training and testing
        X_train = [seq['image_paths'] for seq in self.sequences if seq['patient_id'] in X_train_ids]
        X_test = [seq['image_paths'] for seq in self.sequences if seq['patient_id'] in X_test_ids]
        y_train = [seq['label'] for seq in X_train]
        y_test = [seq['label'] for seq in X_test]

        return X_train, X_test, y_train, y_test


    def create_dataloaders(self):
        # Get train test split
        X_train, X_test, y_train, y_test = self.get_train_split()
        # Create dataset instances
        train_dataset = Breast_MRI(X_train, y_train)
        test_dataset = Breast_MRI(X_test, y_test)

        # Only shuffle the training data, num_workers for parallelization
        training_loader = DataLoader(train_dataset, num_workers=4, shuffle=True)
        testing_loader = DataLoader(test_dataset, num_workers=4, shuffle=False)

        return training_loader, testing_loader


