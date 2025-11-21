from sympy import ceiling
from .roi_crop import CropROId
import torch
from monai.data import (DataLoader, PersistentDataset, list_data_collate, create_test_image_3d)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResizeWithPadOrCropd, CropForegroundd, Spacingd,
    EnsureTyped, NormalizeIntensityd, RandRotate90d, RandSpatialCropd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd)
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
import math
import shutil
import joblib
import os
import sys
import pydicom


class DataUtils:
    def __init__(self, batch_size, image_size, spacing, roi_size):
        # Cache directory for MONAI PersistentDataset
        # Caches previous transformations for faster computation
        self.cache_dir = "cache"
        self.data_dir = "breast_mri_dataset/train_split.joblib"
        self.batch_size = batch_size
        self.image_size = image_size
        self.spacing = spacing
        self.roi_size = roi_size
        # self.target_size = (224, 224, 130)
        self.target_size = (128, 128, 50)
        self.margin = 10
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
                keys=['Folder Path'],
                reader='ITKReader',  # Loads Images using ITKReader which handles 3D volume better
                # reader='PydicomReader',
                image_only=False  # Image_only provides metadata for spacing info
            ),
            EnsureChannelFirstd(
                keys=['Folder Path']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropROId(
                keys=['Folder Path'],
                bbox_key='Bounding Box',
                min_size=self.target_size,
                margin=self.margin  # Adds voxels to X and Y position around the ROI
            ),
            ResizeWithPadOrCropd(
                keys=["Folder Path"],
                spatial_size=self.target_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Folder Path"],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
            # Spatial augmentations
            # RandRotate90d(
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     spatial_axes=(0, 1)  # Only rotate in axial plane
            # ),
            # RandFlipd(
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     spatial_axis=0  # Left-right flip
            # ),
            # # Intensity augmentations (helps with scanner variability)
            # RandScaleIntensityd(
            #     keys=["Folder Path"],
            #     factors=0.2,  # ±20% intensity scaling
            #     prob=0.5
            # ),
            #
            # RandShiftIntensityd(
            #     keys=["Folder Path"],
            #     offsets=0.1,  # Small intensity shifts
            #     prob=0.5
            # ),
            #
            # RandGaussianNoised(
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     mean=0.0,
            #     std=0.05  # Small random noise
            # ),
            #
            # RandGaussianSmoothd(  # Random smoothing
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     sigma_x=(0.5, 1.0),
            #     sigma_y=(0.5, 1.0),
            #     sigma_z=(0.5, 1.0)
            # ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["Folder Path", "Features"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            )
        ])
        self.test_transform = Compose([
            # ─────────────────────────────────────────────────────────────
            # STAGE 1: LOADING & BASIC PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            LoadImaged(
                keys=['Folder Path'],
                reader='ITKReader',  # Loads Images using ITKReader which handles 3D volume better
                image_only=False  # Image_only provides metadata for spacing info
            ),
            EnsureChannelFirstd(
                keys=['Folder Path']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            CropForegroundd(
                keys=["Folder Path"],
                source_key="Folder Path",
                margin=5  # Keep small margin for context
            ),
            Spacingd(
                keys=["Folder Path"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            # STEP 2: Handle variable length intelligently
            # ResizeWithPadOrCropd is BETTER than Resized for variable lengths
            # - If depth < 64: Pads with zeros (no distortion)
            # - If depth > 64: Crops from center (minimal loss)
            # - If depth ≈ 64: Does nothing
            ResizeWithPadOrCropd(
                keys=["Folder Path"],
                spatial_size=self.image_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            RandSpatialCropd(
                keys=["Folder Path"],
                roi_size=self.image_size,  # PATCH-BASED SAMPLING (128³)
                random_size=False,
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Folder Path"],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["Folder Path", "Features"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            )
        ])

    def compute_target_size(self, train_data):
        max_height = 0
        max_width = 0
        max_depth = 0
        for train_dict in train_data:
            bounding_box = train_dict['Bounding Box']

            start_row = bounding_box[0]
            end_row = bounding_box[1]
            start_col = bounding_box[2]
            end_col = bounding_box[3]
            start_slice = bounding_box[4]
            end_slice = bounding_box[5]

            height_diff = end_row - start_row
            width_diff = end_col - start_col
            depth_diff = end_slice - start_slice

            if height_diff > max_height:
                max_height = height_diff

            if width_diff > max_width:
                max_width = width_diff

            if depth_diff > max_depth:
                max_depth = depth_diff
        print(f'Target Size: {(max_height, max_width, max_depth)}')
        height = max_height + self.margin
        width = max_width + self.margin
        depth = max_depth + self.margin
        print(f'Target Size w Margin: {height, width, depth}')
        target_height = ceiling(height / 8) * 8
        target_width = ceiling(width / 8) * 8
        target_depth = ceiling(depth / 8) * 8
        print(f'Final Target Size: {target_height, target_width, target_depth}')

    def compute_weights(self):
        # Load sequence dictionary Jupyter Notebook
        X_train, _, _ = joblib.load(self.data_dir)
        # Collect Labels
        labels = [X['Label'] for X in X_train]
        # Label Arrays
        T_label = [t for t, n, m in labels]
        N_label = [n for t, n, m in labels]
        M_label = [m for t, n, m in labels]
        # Unique Classes
        T_classes = np.unique(T_label)
        N_classes = np.unique(N_label)
        M_classes = np.unique(M_label)
        # Label Weights
        T_weights = compute_class_weight(class_weight='balanced', classes=T_classes, y=T_label)
        N_weights = compute_class_weight(class_weight='balanced', classes=N_classes, y=N_label)
        M_weights = compute_class_weight(class_weight='balanced', classes=M_classes, y=M_label)
        # Return tensor weights
        return (torch.tensor(T_weights, dtype=torch.float32),
                torch.tensor(N_weights, dtype=torch.float32),
                torch.tensor(M_weights, dtype=torch.float32))

    def get_train_split(self):
        # Load sequence dictionary Jupyter Notebook
        X_train, X_val, X_test = joblib.load(self.data_dir)
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


