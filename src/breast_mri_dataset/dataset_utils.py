from .dataset import Breast_MRI
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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

        # Get all UNIQUE patient ids
        patient_seen = set()
        patient_ids = []
        for patient in self.sequences:
            pid = patient['patient_id']
            if pid not in patient_seen:
                patient_ids.append(pid)
                patient_seen.add(pid)

        # Create a mapping from patient_id to label
        pid_to_label = {patient['patient_id']: patient['label'] for patient in self.sequences}

        # Now get labels in the same order as patient_seen
        labels = [pid_to_label[pid] for pid in patient_seen]

        print(patient_ids)
        print(len(patient_ids))
        print(len(labels))

        print(labels)
        negative = 0
        t_count = 0
        n_count = 0
        m_count = 0
        for label in labels:
            if label[0] > 0:
                t_count += 1
            if label[1] > 0:
                n_count += 1
            if label[2] > 0:
                m_count += 1
            if label[2] == 0.5:
                negative +=1

        print(f'T: {t_count}')
        print(f'N: {n_count}')
        print(f'M: {m_count}')
        print(f'Unknown M: {negative}')

        # Extract T stage (first element of each tuple) for stratification
        T_labels = [label[0] for label in labels]

        # Train Test Split
        X_train_ids, X_test_ids, y_train, y_test = train_test_split(
            patient_ids,
                    labels,
                    test_size=0.20,
                    stratify=T_labels,
                    shuffle=True,
                    random_state=30)

        # Currently list of patient ids
        # Want list of sequences
        # Filter sequences for training and testing
        X_train = [seq['image_paths'] for seq in self.sequences if seq['patient_id'] in X_train_ids]
        X_test = [seq['image_paths'] for seq in self.sequences if seq['patient_id'] in X_test_ids]
        y_train = [seq['label'] for seq in self.sequences if seq['patient_id'] in X_train_ids]
        y_test = [seq['label'] for seq in self.sequences if seq['patient_id'] in X_train_ids]

        # Pad sequences to be of uniform length
        # X_train, X_test = pad_sequence(X_train, batch_first=True), pad_sequence(X_test, batch_first=True)

        return X_train, X_test, y_train, y_test

    def create_datasets(self):
        # Get train test split
        X_train, X_test, y_train, y_test = self.get_train_split()
        # Create dataset instances
        train_dataset = Breast_MRI(X_train, y_train)
        test_dataset = Breast_MRI(X_test, y_test)

        return train_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, test_dataset = self.create_datasets()
        # Only shuffle the training data, num_workers for parallelization
        training_loader = DataLoader(train_dataset, num_workers=4, shuffle=True)
        testing_loader = DataLoader(test_dataset, num_workers=4, shuffle=False)

        return training_loader, testing_loader


