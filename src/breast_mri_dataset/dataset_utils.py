import torch
from monai.data import (DataLoader, PersistentDataset, list_data_collate, create_test_image_3d)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResizeWithPadOrCropd, CropForegroundd, Spacingd,
    EnsureTyped, NormalizeIntensityd, RandRotate90d, RandSpatialCropd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd)
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil
import joblib
import os


def load_sequences_dict():
    # Load sequence dictionary Jupyter Notebook
    # sequences = joblib.load("breast_mri_dataset/sequence_data.joblib")
    sequences = joblib.load("breast_mri_dataset/subset_sequence_data.joblib")
    return sequences

class DataUtils:
    def __init__(self, batch_size, image_size, spacing, roi_size):
        # Cache directory for MONAI PersistentDataset
        # Caches previous transformations for faster computation
        self.cache_dir = "cache"
        self.sequences = load_sequences_dict()
        self.batch_size = batch_size
        self.image_size = image_size
        self.spacing = spacing
        self.roi_size = roi_size
        self.n_splits = 5
        #  Multiprocessing
        self.num_workers = 1
        #  "Transforms a list of dictionaries of tensors into a list of dictionaries
        #  of tensors that have the same size to appease DataLoader"
        # self.collate_fn = PadListDataCollate(mode="constant", constant_values=(-1,))
        self.collate_fn = list_data_collate   # https://github.com/Project-MONAI/MONAI/issues/6279
        # =================================================================
        # TRAINING TRANSFORMS
        # =================================================================
        self.train_transform = Compose([
            # ─────────────────────────────────────────────────────────────
            # STAGE 1: LOADING & BASIC PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            LoadImaged(
                keys=['image_paths'],
                reader='ITKReader',  # Loads Images using ITKReader which handles 3D volume better
                image_only=False  # Image_only provides metadata for spacing info
            ),
            EnsureChannelFirstd(
                keys=['image_paths']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropForegroundd(
                keys=["image_paths"],
                source_key="image_paths",
                margin=5  # Keep small margin for context
            ),
            Spacingd(
                keys=["image_paths"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            ResizeWithPadOrCropd(
                keys=["image_paths"],
                spatial_size=self.image_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            RandSpatialCropd(
                keys=["image_paths"],
                roi_size=self.roi_size,  # PATCH-BASED SAMPLING (128³)
                random_size=False,
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["image_paths"],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
            # Spatial augmentations
            RandRotate90d(
                keys=["image_paths"],
                prob=0.5,
                spatial_axes=(0, 1)  # Only rotate in axial plane
            ),

            RandFlipd(
                keys=["image_paths"],
                prob=0.5,
                spatial_axis=0  # Left-right flip
            ),
            # Intensity augmentations (helps with scanner variability)
            RandScaleIntensityd(
                keys=["image_paths"],
                factors=0.2,  # ±20% intensity scaling
                prob=0.5
            ),

            RandShiftIntensityd(
                keys=["image_paths"],
                offsets=0.1,  # Small intensity shifts
                prob=0.5
            ),

            RandGaussianNoised(
                keys=["image_paths"],
                prob=0.3,
                mean=0.0,
                std=0.05  # Small random noise
            ),

            RandGaussianSmoothd(  # Random smoothing
                keys=["image_paths"],
                prob=0.3,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["image_paths", "features"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["label"],
                dtype=torch.long,
                track_meta=False
            )
        ])
        self.test_transform = Compose([
            # ─────────────────────────────────────────────────────────────
            # STAGE 1: LOADING & BASIC PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            LoadImaged(
                keys=['image_paths'],
                reader='ITKReader',  # Loads Images using ITKReader which handles 3D volume better
                image_only=False  # Image_only provides metadata for spacing info
            ),
            EnsureChannelFirstd(
                keys=['image_paths']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropForegroundd(
                keys=["image_paths"],
                source_key="image_paths",
                margin=5  # Keep small margin for context
            ),
            Spacingd(
                keys=["image_paths"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            # STEP 2: Handle variable length intelligently
            # ResizeWithPadOrCropd is BETTER than Resized for variable lengths
            # - If depth < 64: Pads with zeros (no distortion)
            # - If depth > 64: Crops from center (minimal loss)
            # - If depth ≈ 64: Does nothing
            ResizeWithPadOrCropd(
                keys=["image_paths"],
                spatial_size=self.image_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["image_paths"],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["image_paths", "features"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["label"],
                dtype=torch.long,
                track_meta=False
            )
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
        # Reset cache directory
        if os.path.exists(self.cache_dir):
            print('Clearing Cache Directory')
            # shutil.rmtree(self.cache_dir)

        # Get train test split
        X_train, X_test = self.get_train_split()
        # Create dataset instances
        train_dataset = PersistentDataset(data=X_train, transform=self.train_transform,
                                          cache_dir=self.cache_dir)
        test_dataset = PersistentDataset(data=X_test, transform=self.test_transform,
                                         cache_dir=self.cache_dir)
        return train_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, test_dataset = self.create_datasets()
        # Only shuffle the training data, num_workers for parallelization
        training_loader = DataLoader(train_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     pin_memory=torch.cuda.is_available(),
                                     collate_fn=None,
                                    persistent_workers=True
                                     )
        testing_loader = DataLoader(test_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=torch.cuda.is_available(),
                                    collate_fn=None,
                                    persistent_workers=True
                                    )
        return training_loader, testing_loader


