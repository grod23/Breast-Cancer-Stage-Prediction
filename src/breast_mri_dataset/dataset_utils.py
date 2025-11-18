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
import sys


class DataUtils:
    def __init__(self, batch_size, image_size, spacing, roi_size):
        # Cache directory for MONAI PersistentDataset
        # Caches previous transformations for faster computation
        self.cache_dir = "cache"
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
            # RandRotate90d(
            #     keys=["image_paths"],
            #     prob=0.5,
            #     spatial_axes=(0, 1)  # Only rotate in axial plane
            # ),
            #
            # RandFlipd(
            #     keys=["image_paths"],
            #     prob=0.5,
            #     spatial_axis=0  # Left-right flip
            # ),
            # # Intensity augmentations (helps with scanner variability)
            # RandScaleIntensityd(
            #     keys=["image_paths"],
            #     factors=0.2,  # ±20% intensity scaling
            #     prob=0.5
            # ),
            #
            # RandShiftIntensityd(
            #     keys=["image_paths"],
            #     offsets=0.1,  # Small intensity shifts
            #     prob=0.5
            # ),
            #
            # RandGaussianNoised(
            #     keys=["image_paths"],
            #     prob=0.3,
            #     mean=0.0,
            #     std=0.05  # Small random noise
            # ),
            #
            # RandGaussianSmoothd(  # Random smoothing
            #     keys=["image_paths"],
            #     prob=0.3,
            #     sigma_x=(0.5, 1.0),
            #     sigma_y=(0.5, 1.0),
            #     sigma_z=(0.5, 1.0)
            # ),
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
            RandSpatialCropd(
                keys=["image_paths"],
                roi_size=self.image_size,  # PATCH-BASED SAMPLING (128³)
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
        # Load sequence dictionary Jupyter Notebook
        X_train, X_val, X_test = joblib.load("breast_mri_dataset/train_split.joblib")
        return X_train, X_val, X_test

    def create_datasets(self):
        # Reset cache directory
        if os.path.exists(self.cache_dir):
            print('Clearing Cache Directory')
            # shutil.rmtree(self.cache_dir)

        # Get train test split
        X_train, X_val, X_test = self.get_train_split()
        # Create dataset instances
        train_dataset = PersistentDataset(data=X_train, transform=self.train_transform,
                                          cache_dir=self.cache_dir)
        validation_dataset = PersistentDataset(data=X_val, transform=self.test_transform,
                                          cache_dir=self.cache_dir)
        test_dataset = PersistentDataset(data=X_test, transform=self.test_transform,
                                         cache_dir=self.cache_dir)
        return train_dataset, validation_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, validation_dataset, test_dataset = self.create_datasets()
        # Only shuffle the training data, num_workers for parallelization
        training_loader = DataLoader(train_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     pin_memory=torch.cuda.is_available(),
                                     collate_fn=None,
                                    persistent_workers=True
                                     )
        validation_loader = DataLoader(validation_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
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
        return training_loader, validation_loader, testing_loader


