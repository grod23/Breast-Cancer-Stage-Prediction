from .transform_functions import CropROId
from .transform_functions import ScaleIntensity
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, EnsureTyped, NormalizeIntensityd, Resized,
    DeleteItemsd, RandRotate90d,  RandFlipd, ScaleIntensityRanged, RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd)

# Image Preprocessing
class Transform:
    def __init__(self, image_size, spacing, margin):
        self.image_size = image_size
        self.spacing = spacing
        self.margin = margin
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
            CropROId(
                keys=['Folder Path'],
                bbox_key='Bounding Box',
                margin=self.margin,  # Adds voxels to X and Y position around the ROI
            ),
            Spacingd(
                keys=["Window"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            Resized(
                keys=["Window"],
                spatial_size=self.image_size
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Window"],
                nonzero=True,
                channel_wise=False
            ),
            ScaleIntensity(
                keys=["Window"]
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
            #     prob=0.2
            # ),
            # RandShiftIntensityd(
            #     keys=["Folder Path"],
            #     offsets=0.1,  # Small intensity shifts
            #     prob=0.2
            # ),
            # RandGaussianNoised(
            #     keys=["Folder Path"],
            #     prob=0.2,
            #     mean=0.0,
            #     std=0.05  # Small random noise
            # ),
            # RandGaussianSmoothd(  # Random smoothing
            #     keys=["Folder Path"],
            #     prob=0.2,
            #     sigma_x=(0.5, 1.0),
            #     sigma_y=(0.5, 1.0),
            #     sigma_z=(0.5, 1.0)
            # ),
            # # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["Window", "Features"],
                dtype=torch.float32,
                track_meta=False,
                allow_missing_keys=True
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            ),
            # Remove metadata
            DeleteItemsd(
                keys=['Folder Path_meta_dict', 'Folder Path']
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
            CropROId(
                keys=['Folder Path'],
                bbox_key='Bounding Box',
                margin=self.margin,  # Adds voxels to X and Y position around the ROI
                deterministic=True
            ),
            Spacingd(
                keys=["Window"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            Resized(
                keys=["Window"],
                spatial_size=self.image_size
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Window"],
                nonzero=True,
                channel_wise=False
            ),
            ScaleIntensity(
                keys=["Window"]
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["Window", "Features"],
                dtype=torch.float32,
                track_meta=False,
                allow_missing_keys=True
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            ),
            # Remove metadata
            DeleteItemsd(
                keys=['Folder Path_meta_dict', 'Folder Path']
            )
        ])