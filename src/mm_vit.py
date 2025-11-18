import torch
import torch.nn as nn
from monai.networks.nets import DenseNet, UNet, UNETR, SwinUNETR, ViTAutoEnc, VISTA3D, DenseNet121
from typing import Tuple, Optional
import torch.nn.functional as F
# from tests.networks.nets.test_milmodel import pretrained

#     Architecture:
#         1. 3D Vision Transformer (ViT) processes MRI volumes independently
#         2. MLP processes clinical features independently
#         3. Late fusion combines embeddings at decision level
#         4. Final classification head

# Since labels is multioutput: (T, N, M)
# Treat each label as its own input and output

# Tabular Multi Layer Perceptron
class MLP_Encoder(nn.Module):
    def __init__(self, input_dim=19, output_dim=128):  # For now, we have 19 features
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, X):
        X = self.encoder(X)  # Shape: ([3, 1, 128])
        X = X.squeeze(1)  # Shape: ([3, 128])
        return X

# Medical 3D Vision Transformer
class DenseNetEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        # Initialize DenseNet for 3D classification
        self.densenet = DenseNet121(
            spatial_dims=3,  # 3D volumes
            in_channels=in_channels,
            out_channels=hidden_size,
            pretrained=False
        )

        # DenseNet121 has 1024 features, DenseNet169 has 1664
        feature_dims = 1024
        densenet_features = feature_dims

        # Project to desired hidden size
        self.projection = nn.Sequential(
            nn.Linear(densenet_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, X):
        # Extract features from DenseNet (before final classification layer)
        features = self.densenet.features(X)

        # Global Average Pooling: [B, C, H, W, D] -> [B, C]
        pooled = F.adaptive_avg_pool3d(features, (1, 1, 1))
        flattened = pooled.view(pooled.size(0), -1)

        # Project to hidden_size
        projected = self.projection(flattened)

        return projected

    def load_model_weights(self):
        # https: // github.com / Project - MONAI / tutorials / blob / main / self_supervised_pretraining / vit_unetr_ssl / ssl_finetune.ipynb
        pretrained_path = ''
        print("Loading Weights from the Path {}".format(pretrained_path))
        vit_dict = torch.load(pretrained_path, weights_only=True)
        vit_weights = vit_dict['state_dict']
        # " Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
        # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
        # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
        # while pretraining with ViTAutoEnc and are not a part of ViT backbone".
        model_dict = self.unet.vit.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        self.unet.vit.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict
        print("Pretrained Weights Loaded!")

class MultiModalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = DenseNetEncoder()
        self.tabular_encoder = MLP_Encoder()

        image_hidden = self.image_encoder.hidden_size
        tabular_hidden = self.tabular_encoder.output_dim

        # MLP for modality fusion
        self.fusion_head = nn.Sequential(
            nn.Linear(image_hidden + tabular_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Each label has different output classes
        self.output_T = nn.Linear(in_features=128, out_features=5)
        self.output_N = nn.Linear(in_features=128, out_features=4)
        self.output_M = nn.Linear(in_features=128, out_features=2)

    def forward(self, X_images, X_features):
        image_features= self.image_encoder(X_images)
        tabular_features = self.tabular_encoder(X_features)

        # print(f'Image Features: {image_features}')
        # print(f'Image Features Shape: {image_features.shape}')
        # print(f'Tabular Features Shape: {tabular_features.shape}')

        combined = torch.cat([image_features, tabular_features], dim=1)
        y = self.fusion_head(combined)
        # Separately get output for each label
        T_out = self.output_T(y)
        N_out = self.output_N(y)
        M_out = self.output_M(y)
        return T_out, N_out, M_out
