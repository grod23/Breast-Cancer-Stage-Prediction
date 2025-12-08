import torch
import torch.nn as nn
from monai.networks.nets import ResNet
from torchvision.models import resnet50, densenet121, inception_v3

resnet_path = r"C:\Users\gabe7\Downloads\RadImageNet_pytorch\RadImageNet_pytorch\ResNet50.pt"

'''
@article{doi:10.1148/ryai.210315,
author = {Mei, Xueyan and Liu, Zelong and Robson, Philip M. and Marinelli, Brett and Huang, Mingqian and Doshi, 
Amish and Jacobi, Adam and Cao, Chendi and Link, Katherine E. and Yang, Thomas and Wang, Ying and Greenspan, 
Hayit and Deyer, Timothy and Fayad, Zahi A. and Yang, Yang},
title = {RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning},
journal = {Radiology: Artificial Intelligence},
volume = {0},
number = {ja},
pages = {e210315},
year = {0},
doi = {10.1148/ryai.210315},

URL = { 
        https://doi.org/10.1148/ryai.210315
    
},
eprint = { 
        https://doi.org/10.1148/ryai.210315
}
}
'''


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)

class MLPEncoder(nn.Module):
    def __init__(self, output_dim, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, output_dim)
        )

    def forward(self, X_features):
        return self.mlp(X_features)

class SequenceEncoder25D(nn.Module):
    def __init__(self, output_dim, in_channels=3, aggregation='attention'):
        super().__init__()
        self.block_inplanes = 128
        self.aggregation = aggregation
        self.backbone = ResNet(
            block='basic',
            n_input_channels=in_channels,
            spatial_dims=2,
            layers=[2, 2, 2, 2],
            # layers=[1, 1, 1, 1],
            block_inplanes = [16, 32, 64, self.block_inplanes],
            feed_forward=False
        )
        # Get output features from last stage
        # num_features = 2048  # ResNet50
        # self.backbone = Backbone()
        # Load ResNet50 weights
        # self.backbone.load_state_dict(torch.load(resnet_path))

        # Replace classifier with feature extractor
        # self.backbone.fc = nn.Identity()

        # self.feature_head = nn.Sequential(
        #     nn.Linear(self.block_inplanes, output_dim),
        # )
        # self.feature_head = nn.Sequential(
        #     nn.Linear(self.block_inplanes, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, output_dim)
        # )

    def forward(self, X_images):
        features = self.backbone(X_images)
        output = features
        return output

class MultiModalClassifier(nn.Module):
    def __init__(self, fusion_strategy, fusion_dim, num_classes=4, num_clinical_features=5):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        # Encoders
        self.image_encoder = SequenceEncoder25D(
            output_dim=fusion_dim,
            in_channels=3
        )
        self.feature_encoder = MLPEncoder(
            output_dim=fusion_dim,
            num_features=num_clinical_features
        )
        # Fusion layer
        if fusion_strategy == 'concat':
            # Simple concatenation
            fusion_input_dim = fusion_dim * 2
        elif fusion_strategy == 'bilinear':
            # Bilinear pooling for richer interactions
            self.bilinear = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
            fusion_input_dim = fusion_dim
        elif fusion_strategy == 'cross_attention':
            # Cross-attention between modalities
            self.cross_attention = CrossModalAttention(fusion_dim)
            fusion_input_dim = fusion_dim * 2
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        # Final classifier
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, X_images, X_features):
        # Encode both modalities
        image_features = self.image_encoder(X_images)
        clinical_features = self.feature_encoder(X_features)
        # Fuse features
        if self.fusion_strategy == 'concat':
            combined_features = torch.cat([image_features, clinical_features], dim=1)
        elif self.fusion_strategy == 'bilinear':
            combined_features = self.bilinear(image_features, clinical_features)
        elif self.fusion_strategy == 'cross_attention':
            attended_img, attended_clin = self.cross_attention(image_features, clinical_features)
            combined_features = torch.cat([attended_img, attended_clin], dim=1)

        # Final classification
        output = self.fusion_mlp(combined_features)
        return output

class CrossModalAttention(nn.Module):
    # Vaswani et al. (2017) "Attention Is All You Need"
    def __init__(self, dim):
        super().__init__()
        self.query_img = nn.Linear(dim, dim)
        self.key_clin = nn.Linear(dim, dim)
        self.value_clin = nn.Linear(dim, dim)

        self.query_clin = nn.Linear(dim, dim)
        self.key_img = nn.Linear(dim, dim)
        self.value_img = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, img_features, clin_features):
        # Image attends to clinical
        q_img = self.query_img(img_features).unsqueeze(1)  # [B, 1, D]
        k_clin = self.key_clin(clin_features).unsqueeze(1)
        v_clin = self.value_clin(clin_features).unsqueeze(1)

        attn_img = torch.softmax(q_img @ k_clin.transpose(-2, -1) * self.scale, dim=-1)
        attended_img = (attn_img @ v_clin).squeeze(1) + img_features

        # Clinical attends to image
        q_clin = self.query_clin(clin_features).unsqueeze(1)
        k_img = self.key_img(img_features).unsqueeze(1)
        v_img = self.value_img(img_features).unsqueeze(1)

        attn_clin = torch.softmax(q_clin @ k_img.transpose(-2, -1) * self.scale, dim=-1)
        attended_clin = (attn_clin @ v_img).squeeze(1) + clin_features

        return attended_img, attended_clin
