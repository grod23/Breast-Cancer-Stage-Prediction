import torch
import monai
from monai.data import (DataLoader, Dataset, CacheDataset, PersistentDataset,
                        create_test_image_3d, list_data_collate, decollate_batch)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    CropForegroundd, ScaleIntensityRanged, Spacingd, Resized,
    NormalizeIntensityd, ToTensord, Activationsd, AsDiscreted,
    RandCropByPosNegLabeld, RandRotate90d, ScaleIntensityd,)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
import sys

def load_sequences_dict():
    # Load
    sequences = joblib.load("breast_mri_dataset/sequence_data.joblib")
    return sequences

class DataUtils:
    def __init__(self):
        self.sequences = load_sequences_dict()
        self.target_size = (64,128,128)
        self.spacing = (1.0, 1.0, 1.0)
        self.train_transform = Compose([
            LoadImaged(keys=['image_paths']),  # Loads Images
            EnsureChannelFirstd(keys=['image_paths']),  # Ensures correct chanel format (Channels, Depth, Height, Width)
            Spacingd(keys=["image_paths"], pixdim=self.spacing, mode="bilinear"),  # Ensures consistent voxel spacing
            Resized(keys=["image_paths"], spatial_size=self.target_size),  # Ensures consistent image size
            NormalizeIntensityd(keys=["image_paths"], nonzero=True, channel_wise=False),  # Z-Score Standardization
            ToTensord(keys=["image_paths", "label", "features"])  # Converts Images, Labels, and Features to tensor
            # ScaleIntensityd(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0),
        ])

        self.test_transform = Compose([
            LoadImaged(keys=['image_paths']),
            EnsureChannelFirstd(keys=['image_paths']),
            # CropForegroundd(keys=['image'], select_fn=lambda x: x > 0),
            Spacingd(keys=["image_paths"], pixdim=self.spacing, mode="bilinear"),
            Resized(keys=["image_paths"], spatial_size=self.target_size),
            NormalizeIntensityd(keys=["image_paths"], nonzero=True, channel_wise=False),
            ToTensord(keys=["image_paths", "label", "features"])
        ])

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

        print(f'Patient IDs Length: {len(patient_ids)}')
        print(f'Labels Length: {len(labels)}')

        # Count combined TNM distributions
        combined_counts = Counter(labels)
        print("Combined (T,N,M) distribution:")
        for combo, count in combined_counts.most_common():
            print(f"{combo}: {count}")

        # Count T, N, M individually
        t_counts = Counter()
        n_counts = Counter()
        m_counts = Counter()

        for t, n, m in labels:
            t_counts[t] += 1
            n_counts[n] += 1
            m_counts[m] += 1

        print("T-stage distribution:", t_counts)
        print("N-stage distribution:", n_counts)
        print("M-stage distribution:", m_counts)

        # Extract T stage (first element of each tuple) for stratification
        T_labels = [label[0] for label in labels]

        # Train Test Split
        X_train_ids, X_test_ids = train_test_split(
            patient_ids,
            test_size=0.2,
            stratify=T_labels,  # ensures class distribution of T-stage
            shuffle=True,
            random_state=30
        )

        # self.sequences is list of dictionary patients
        # Train Test split currently list of patient ID's
        # MONAI needs list of dictionary sequences

        # sequence{
        # 'patient_id':
        # 'image_paths':
        # 'label':
        # }

        X_train = [patient_dict for patient_dict in self.sequences
                   if patient_dict['patient_id'] in X_train_ids]

        X_test = [patient_dict for patient_dict in self.sequences
                   if patient_dict['patient_id'] in X_test_ids]

        # May contain empty lists (Remove empty lists)
        X_train = [patient_dict for patient_dict in X_train if patient_dict]
        X_test = [patient_dict for patient_dict in X_test if patient_dict]

        # y_train = [list(seq['label']) for seq in self.sequences if seq['patient_id'] in X_train_ids]
        # y_test = [list(seq['label']) for seq in self.sequences if seq['patient_id'] in X_test_ids]

        return X_train, X_test

    def create_datasets(self):
        # Get train test split
        X_train, X_test = self.get_train_split()
        # Create dataset instances
        train_dataset = Dataset(data=X_train, transform=self.train_transform)
        test_dataset = Dataset(data=X_test, transform=self.test_transform)

        return train_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, test_dataset = self.create_datasets()
        # Only shuffle the training data, num_workers for parallelization
        training_loader = DataLoader(train_dataset, batch_size=3, num_workers=4,
                                     shuffle=True, pin_memory=torch.cuda.is_available())
        testing_loader = DataLoader(test_dataset, batch_size=3, num_workers=4,
                                    shuffle=False, pin_memory=torch.cuda.is_available())

        return training_loader, testing_loader


