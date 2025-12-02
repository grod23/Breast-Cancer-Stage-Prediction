from .roi_crop import CropROId
from .spatial_crop import SpatialCropBBoxd
from.scale_intensity import ScaleIntensity
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResizeWithPadOrCropd, CropForegroundd, Spacingd, SpatialCropd,
    EnsureTyped, NormalizeIntensityd, RandRotate90d, RandSpatialCropd, RandFlipd, ScaleIntensityRanged,
    RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd, DeleteItemsd, SpatialPadd, RandGridPatchd,
    RandScaleIntensityd)
from monai.transforms import MapTransform

class Transform:
    def __init__(self, image_size, roi_size, spacing, margin):
        self.image_size = image_size
        self.roi_size = roi_size
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
                image_size=self.image_size,
                margin=self.margin,  # Adds voxels to X and Y position around the ROI
                test=False
            ),
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            # CropForegroundd(
            #     keys=["Folder Path"],
            #     source_key="Folder Path",
            #     margin=5  # Keep small margin for context
            # ),
            Spacingd(
                keys=["Folder Path"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            # RandSpatialCropd(
            #     keys=["Folder Path"],
            #     roi_size=self.roi_size,
            # ),
            # Resize Full Image
            ResizeWithPadOrCropd(
                keys=["Folder Path"],
                spatial_size=self.image_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            # Resize ROI Crop
            # ResizeWithPadOrCropd(
            #     keys=["ROI Crop"],
            #     spatial_size=self.roi_size,
            #     mode='edge',  # 'edge' mode pads by repeating edge values
            # ),
            SpatialCropBBoxd(
                target_size=self.roi_size,
                keys=["ROI Crop"],
                bbox_key='Bounding Box'
            ),
            SpatialPadd(
                keys=["ROI Crop"],
                spatial_size=self.roi_size,
                method='symmetric',
                mode='constant'
                        ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Folder Path"],
                nonzero=True,
                channel_wise=False
            ),
            ScaleIntensity(
                keys=["Folder Path", "ROI Crop"]
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
            # Spatial augmentations
            RandRotate90d(
                keys=["Folder Path"],
                prob=0.3,
                spatial_axes=(0, 1)  # Only rotate in axial plane
            ),
            RandFlipd(
                keys=["Folder Path"],
                prob=0.3,
                spatial_axis=0  # Left-right flip
            ),
            # Intensity augmentations (helps with scanner variability)
            RandScaleIntensityd(
                keys=["Folder Path"],
                factors=0.2,  # ±20% intensity scaling
                prob=0.5
            ),
            RandShiftIntensityd(
                keys=["Folder Path"],
                offsets=0.1,  # Small intensity shifts
                prob=0.5
            ),
            RandGaussianNoised(
                keys=["Folder Path"],
                prob=0.3,
                mean=0.0,
                std=0.05  # Small random noise
            ),
            RandGaussianSmoothd(  # Random smoothing
                keys=["Folder Path"],
                prob=0.3,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            # DecrementLabeld(
            #     keys=['Label']
            # ),
            EnsureTyped(
                keys=["Folder Path", "Features", "ROI Crop"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            ),
            # Remove metadata
            DeleteItemsd(
                keys=['Folder Path_meta_dict']
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
                image_size=self.image_size,
                margin=self.margin,  # Adds voxels to X and Y position around the ROI
                test=False
            ),
            # Remove background slices (reduces variability)
            # This crops empty slices at start/end of volume
            # CropForegroundd(
            #     keys=["Folder Path"],
            #     source_key="Folder Path",
            #     margin=5  # Keep small margin for context
            # ),
            Spacingd(
                keys=["Folder Path"],
                pixdim=self.spacing,  # Standardize voxel spacing
                mode="bilinear"
            ),
            # RandSpatialCropd(
            #     keys=["Folder Path"],
            #     roi_size=self.roi_size,
            # ),
            # Resize Full Image
            ResizeWithPadOrCropd(
                keys=["Folder Path"],
                spatial_size=self.image_size,
                mode='edge'  # 'edge' mode pads by repeating edge values
            ),
            # Resize ROI Crop
            # ResizeWithPadOrCropd(
            #     keys=["ROI Crop"],
            #     spatial_size=self.roi_size,
            #     mode='edge',  # 'edge' mode pads by repeating edge values
            # ),
            SpatialCropBBoxd(
                target_size=self.roi_size,
                keys=["ROI Crop"],
                bbox_key='Bounding Box'
            ),
            SpatialPadd(
                keys=["ROI Crop"],
                spatial_size=self.roi_size,
                method='symmetric',
                mode='constant'
                        ),
            # Localization patches
            # RandSpatialCropd(
            #     keys=['ROI Crop'],
            #     roi_size=self.roi_size,
            #     random_center=True,
            #     random_size=False
            # ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=["Folder Path"],
                nonzero=True,
                channel_wise=False
            ),
            ScaleIntensity(
                keys=["Folder Path", "ROI Crop"]
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=["Folder Path", "Features", "ROI Crop"],
                dtype=torch.float32,
                track_meta=False
            ),
            EnsureTyped(
                keys=["Label"],
                dtype=torch.long,
                track_meta=False
            ),
            # Remove metadata
            DeleteItemsd(
                keys=['Folder Path_meta_dict']
            )
        ])


