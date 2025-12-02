import torch
import torch.nn as nn
from monai.networks.nets import (DenseNet, UNet, UNETR, SwinUNETR, ViTAutoEnc, HighResNet,
                                 VISTA3D, DenseNet121, ViT, ResNet, Densenet121)
import torch
import torch.nn.functional as F
# from tests.networks.nets.test_milmodel import pretrained
import sys

#     Architecture:
#         1. 3D Vision Transformer (ViT) processes MRI volumes independently
#         2. MLP processes clinical features independently
#         3. Late fusion combines embeddings at decision level
#         4. Final classification head

# Since labels is multioutput: (T, N, M)
# Treat each label as its own input and output

class ROIBranch(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.backbone = HighResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_dim,
            dropout_prob=0.2
        )
        # Add global pooling to convert spatial features to fixed vector
        self.global_pool = nn.AdaptiveAvgPool3d(1)
    def forward(self, X):
        # X: [B, 1, H, W, D]
        # print(f'ROI Input: {X.shape}')
        roi_features = self.backbone(X)  # [B, 256, H', W', D']
        # print(f'ROI Features (spatial): {roi_features.shape}')
        # Pool to fixed size
        pooled = self.global_pool(roi_features)  # [B, 256, 1, 1, 1]
        pooled_features = pooled.view(pooled.size(0), -1)  # [B, 256]
        # print(f'ROI Features (pooled): {pooled_features.shape}\n')
        return pooled_features  # [B, feature_dim]


class VisionTransformer(nn.Module):
    def __init__(self, image_size, in_channels=1, feature_dim=256):
        super().__init__()

        self.backbone = ViT(
            spatial_dims=3,
            in_channels=in_channels,
            img_size=image_size,
            patch_size=(16, 16, 16),
            # hidden_size=feature_dim,
            # num_heads=12,
            dropout_rate=0.2,
            classification=False
        )
        self.projection = nn.Linear(768, feature_dim)

    def forward(self, X):
        # print(f'Input Sequence: {X.shape}')
        sequence_features = self.backbone(X)[1] # [12, B, feature_dim, hidden_size]
        # Sequence features is a list of layers
        # Retrieve last layers CLS token
        last_layer = sequence_features[-1]  # Shape: [1, 256, 768]
        # print(f'Last Layer: {last_layer.shape}')
        cls_token = last_layer[:, 0]  # Shape: [1, 768]
        # print(f'CLS Token: {cls_token.shape}')
        output = self.projection(cls_token)
        # print(f'ViT Output: {output.shape}')
        return output


class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, roi_features, global_features):
        # print(f'Local Features: {roi_features.shape}')
        # print(f'Global Features: {global_features.shape}')
        # Both should now be [B, embed_dim]
        # Stack into sequence for attention
        X = torch.stack([roi_features, global_features], dim=1)  # [B, 2, embed_dim]
        # print(f'Stacked Features: {X.shape}')
        # Self-attention over the two feature vectors
        X2, _ = self.attn(X, X, X)  # [B, 2, embed_dim]
        # print(f'Attention Features: {X2.shape}')
        # Residual connection and normalization
        X = self.norm(X + X2)
        # print(f'LayerNorm Features: {X.shape}')
        # Pool the two attended features
        X = X.mean(dim=1)  # [B, embed_dim]
        # print(f'Attention Output: {X.shape}\n')
        return X

class MultiscaleClassifier(nn.Module):
    def __init__(self, image_size, in_channels=1, feature_dim=256, num_classes=4):
        super().__init__()
        self.roi_branch = ROIBranch(in_channels, feature_dim)
        self.full_branch = VisionTransformer(image_size, in_channels, feature_dim)
        self.hier_attn = HierarchicalAttention(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, sequence, roi_patch):
        # print(f'ROI Patch Shape: {roi_patch.shape}')
        # print(f'Full Volume Shape: {sequence.shape}')
        roi_features = self.roi_branch(roi_patch)
        global_features = self.full_branch(sequence)
        # print(f'Model ROI Features: {roi_features.shape}')
        # print(f'Model Global Features: {global_features.shape}')
        fusion_features = self.hier_attn(roi_features, global_features)
        # print(f'Model Fusion Features: {fusion_features.shape}')
        out = self.classifier(fusion_features)
        # print(f'Model Logits Shape: {out.shape}')
        # print(f'Model Logits: {out}\n')
        return out


class MLPEncoder(nn.Module):
    def __init__(self, num_features, output_dim = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Linear(128, output_dim)
        )