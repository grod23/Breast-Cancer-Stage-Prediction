import pydicom
from monai.data import PydicomReader
import torch
import monai
from monai.data import (DataLoader, Dataset, CacheDataset, PersistentDataset,
                        create_test_image_3d, list_data_collate, decollate_batch)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResizeWithPadOrCropd, Orientationd,
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
    # Load sequence dictionary Jupyter Notebook
    sequences = joblib.load("breast_mri_dataset/sequence_data.joblib")
    return sequences

class DataUtils:
    def __init__(self):
        self.sequences = load_sequences_dict()
        self.target_size = (64,128,128)
        self.spacing = (1.0, 1.0, 1.0)
        self.n_splits = 5
        self.train_transform = Compose([
            # Loads Images using ITKReader which handles 3D volume better. Image_only provides metadata for spacing info
            LoadImaged(
                keys=['image_paths'],
                reader='ITKReader',
                image_only=False
            ),
            # Ensures correct channel format (Channels, Depth, Height, Width)
            EnsureChannelFirstd(
                keys=['image_paths']
            ),
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropForegroundd(
                keys=["image_paths"],
                source_key="image_paths",
                margin=5  # Keep small margin for context
            ),
            # Ensures consistent voxel spacing
            Spacingd(
                keys=["image_paths"],
                pixdim=self.spacing,
                mode="bilinear"
            ),
            # STEP 2: Handle variable length intelligently
            # ResizeWithPadOrCropd is BETTER than Resized for variable lengths
            # - If depth < 64: Pads with zeros (no distortion)
            # - If depth > 64: Crops from center (minimal loss)
            # - If depth ≈ 64: Does nothing
            ResizeWithPadOrCropd(
                keys=["image_paths"],
                spatial_size=self.target_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),

            # Standardize orientation (Axes are flipped/reordered to [R, A, S]. The voxel data is
            # automatically transposed and/or mirrored so the anatomical directions match RAS.)
            # Orientationd(
            #     keys=["image_paths"],
            #     axcodes="RAS"  # Right-Anterior-Superior standard
            # ),

            # Z-Score Normalization (data - mean) / std_dev
            NormalizeIntensityd(
                keys=["image_paths"],
                nonzero=True,
                channel_wise=False
            ),
            # Converts Images, Labels, and Features to tensor
            ToTensord(
                keys=["image_paths", "label", "features"])
            # ScaleIntensityd(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0),
        ])

        self.test_transform = Compose([
            # Loads Images using ITKReader which handles 3D volume better. Image_only provides metadata for spacing info
            LoadImaged(
                keys=['image_paths'],
                reader='ITKReader',
                image_only=False
            ),
            # Ensures correct channel format (Channels, Depth, Height, Width)
            EnsureChannelFirstd(
                keys=['image_paths']
            ),
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropForegroundd(
                keys=["image_paths"],
                source_key="image_paths",
                margin=5  # Keep small margin for context
            ),
            # Ensures consistent voxel spacing
            Spacingd(
                keys=["image_paths"],
                pixdim=self.spacing,
                mode="bilinear"
            ),
            # STEP 2: Handle variable length intelligently
            # ResizeWithPadOrCropd is BETTER than Resized for variable lengths
            # - If depth < 64: Pads with zeros (no distortion)
            # - If depth > 64: Crops from center (minimal loss)
            # - If depth ≈ 64: Does nothing
            ResizeWithPadOrCropd(
                keys=["image_paths"],
                spatial_size=self.target_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),

            # Standardize orientation (Axes are flipped/reordered to [R, A, S]. The voxel data is
            # automatically transposed and/or mirrored so the anatomical directions match RAS.)
            # Orientationd(
            #     keys=["image_paths"],
            #     axcodes="RAS"  # Right-Anterior-Superior standard
            # ),

            # Z-Score Normalization (data - mean) / std_dev
            NormalizeIntensityd(
                keys=["image_paths"],
                nonzero=True,
                channel_wise=False
            ),
            # Converts Images, Labels, and Features to tensor
            ToTensord(
                keys=["image_paths", "label", "features"])
            # ScaleIntensityd(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0),
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
        combined_counts = Counter(tuple(label) for label in labels)
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

        # Only need the dictionaries, MONAI will automatically retrieve labels and features
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

        # May contain empty lists of sequences.
        # Sequences may also contain empty lists of image paths.

        # Removes all empty lists
        X_train = [patient_dict for patient_dict in X_train if patient_dict and patient_dict['image_paths']]
        X_test = [patient_dict for patient_dict in X_test if patient_dict and patient_dict['image_paths']]

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
        training_loader = DataLoader(train_dataset, batch_size=1, num_workers=4,
                                     shuffle=True, pin_memory=torch.cuda.is_available())
        testing_loader = DataLoader(test_dataset, batch_size=1, num_workers=4,
                                    shuffle=False, pin_memory=torch.cuda.is_available())

        return training_loader, testing_loader


