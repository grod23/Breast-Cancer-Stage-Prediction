from.transforms import Transform
from monai.data import (DataLoader, PersistentDataset)
from monai.transforms import PadListDataCollate
from sklearn.utils.class_weight import compute_class_weight
from monai.visualize import matshow3d
import matplotlib.pyplot as plt
import torch
import numpy as np
from sympy import ceiling
import joblib
import os
import pydicom

class DataUtils:
    def __init__(self, batch_size, image_size, roi_size, spacing, margin):
        # Cache directory for MONAI PersistentDataset
        # Caches previous transformations for faster computation
        self.cache_dir = "cache"
        self.data_dir = "breast_mri_dataset/train_split.joblib"
        self.batch_size = batch_size
        self.margin = margin
        self.transform = Transform(image_size=image_size, roi_size=roi_size, spacing=spacing, margin=margin)
        #  Multiprocessing
        self.num_workers = 2
        #  "Transforms a list of dictionaries of tensors into a list of dictionaries
        #  of tensors that have the same size to appease DataLoader"
        self.collate_fn = PadListDataCollate(mode="constant", constant_values=(-1,))
        # self.collate_fn = list_data_collate   # https://github.com/Project-MONAI/MONAI/issues/6279

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
        T_label = [t - 1 for t, n, m in labels]
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
        print(f'Tumor Classes: {T_classes}')
        print(f'Tumor Weights: {T_weights}')
        # Return tensor weights
        # T_weights = [0.84680851, 0.92990654, 0.71071429, 5]
        return (torch.tensor(T_weights, dtype=torch.float32),
                torch.tensor(N_weights, dtype=torch.float32),
                torch.tensor(M_weights, dtype=torch.float32))

    def load_dicom_series(self, folder):
        # load files
        files = [os.path.join(folder, f)
                 for f in os.listdir(folder) if f.endswith('.dcm')]
        # read
        slices = [pydicom.dcmread(f) for f in files]
        # sort by slice order
        slices.sort(key=lambda s: int(s.InstanceNumber))
        # stack into 3D volume
        volume = np.stack([s.pixel_array for s in slices], axis=0)
        return volume

    def get_train_split(self):
        import sys
        # Load sequence dictionary Jupyter Notebook
        X_train, X_val, X_test = joblib.load(self.data_dir)

        def percentage_label_4(split):
            total = len(split)
            count_4 = sum(1 for patient in split if patient['Label'][0] == 4.0)
            return (count_4 / total) * 100
        print("Percentage of label[0] == 4 in X_train: {:.2f}%".format(percentage_label_4(X_train)))
        print("Percentage of label[0] == 4 in X_val: {:.2f}%".format(percentage_label_4(X_val)))
        print("Percentage of label[0] == 4 in X_test: {:.2f}%".format(percentage_label_4(X_test)))
        sys.exit()
        self.compute_target_size(X_train)
        return X_train, X_val, X_test

    def create_datasets(self):
        # Reset cache directory
        if os.path.exists(self.cache_dir):
            print('Clearing Cache Directory')
            # shutil.rmtree(self.cache_dir)

        # Get train test split
        X_train, X_val, X_test = self.get_train_split()
        # Create dataset instances
        train_dataset = PersistentDataset(
                                          data=X_train,
                                          transform=self.transform.train_transform,
                                          cache_dir=self.cache_dir)
        validation_dataset = PersistentDataset(
                                               data=X_val,
                                               transform=self.transform.test_transform,
                                               cache_dir=self.cache_dir)
        test_dataset = PersistentDataset(
                                         data=X_test,
                                         transform=self.transform.test_transform,
                                         cache_dir=self.cache_dir
                                         )
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


