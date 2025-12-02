import torch
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.visualize import matshow3d, blend_images
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Custom MONAI Transform to crop a 3D volumetric image to a specified bounding box.
class CropROId(MapTransform):
    def __init__(self, keys: KeysCollection, image_size, margin, test, bbox_key='Bounding Box'):
        super().__init__(keys)
        self.bbox_key = bbox_key
        self.margin = margin
        self.image_size = image_size
        self.test = test

    def __call__(self, data):
        d = dict(data)  # Create a mutable copy of the input dictionary
        patient = d.get('Patient ID')
        bounding_box = d.get(self.bbox_key)
        if bounding_box is None:
            # If no bbox, return original data (fallback)
            print(f"Warning: No bbox found for patient {d.get('Patient ID', 'unknown')}")
            return d
        for key in self.keys:
            sequence = d[key]
            # Don't do anything if this is testing data
            # if self.test:
            #     d['ROI Crop'] = sequence
            #     return d
            # Image shape: [C, H, W, D] where C=1 for MRI
            # C, H, W, D = sequence.shape
            # Bounding Box Format:
            # [Start_Row, End_Row, Start_Column, End_Column, Start_Slice, End_Slice]
            padding = self.margin / 2
            start_row = max(0, int(bounding_box[0] - padding))   # Clamp to avoid -indices
            end_row = min(sequence.shape[1], int(bounding_box[1] + padding)) # Clamp to avoid out of bounds indices
            start_col = max(0, int(bounding_box[2] - padding))   # Clamp to avoid -indices
            end_col = min(sequence.shape[2], int(bounding_box[3] + padding)) # Clamp to avoid out of bounds indices
            start_slice = max(0, int(bounding_box[4] - padding))  # Clamp to avoid -indices
            end_slice = min(sequence.shape[3], int(bounding_box[5] + padding)) # Clamp to avoid out of bounds indices

            # matshow3d(
            #     title=f'{patient}',
            #     volume=sequence,
            #     every_n=5,
            #     cmap='gray',
            #     figsize=(10, 10)
            # )
            plt.show()
            # Crop sequence
            cropped = sequence[:,
            start_row:end_row,
            start_col:end_col,
            start_slice:end_slice]
            # Store cropped image
            if 0 in cropped.shape:
                print('SKIPPING SEQUENCE')
                print(cropped.shape)
                print(f'Patient: {patient}')
                print(f'Bounding Box: {bounding_box}')
                print(f'Sequence: {sequence.shape}')
                raise Exception('INCORRECT CROPPING')
                d['ROI Crop'] = sequence
                return d

            # Check for cropping
            # if self.image_size[2] < (end_slice - start_slice):
            #     print(f'CROPPING: {(end_slice - start_slice)} to {self.image_size[2]}')
            # matshow3d(
            #     title=f'{patient} Cropped: {[end_row - start_row, end_col - start_col, end_slice - start_slice]}',
            #     volume=cropped,
            #     every_n=1,  # Show every n slice
            #     cmap='gray',
            #     figsize=(10, 10)
            # )
            # plt.show()
            d['ROI Crop'] = cropped

        return d
