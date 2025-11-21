import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.visualize import matshow3d, blend_images
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Custom MONAI Transform to crop a 3D volumetric image to a specified bounding box.
class CropROId(MapTransform):
    def __init__(self, keys: KeysCollection, min_size, margin, bbox_key='Bounding Box'):
        super().__init__(keys)
        self.bbox_key = bbox_key
        self.margin = margin
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)  # Create a mutable copy of the input dictionary
        patient = d.get('Patient ID')
        bounding_box = d.get(self.bbox_key)
        if bounding_box is None:
            # If no bbox, return original data (fallback)
            print(f"Warning: No bbox found for patient {d.get('Patient ID', 'unknown')}")
            return d

        # Keys: ['Folder Path']
        for key in self.keys:
            sequence = d[key]
            # Image shape: [C, H, W, D] where C=1 for MRI
            # C, H, W, D = sequence.shape
            # Bounding Box Format:
            # [Start_Row, End_Row, Start_Column, End_Column, Start_Slice, End_Slice]
            padding = self.margin / 2
            start_row = int(bounding_box[0] - padding)
            end_row = int(bounding_box[1] + padding)
            start_col = int(bounding_box[2] - padding)
            end_col = int(bounding_box[3] + padding)
            start_slice = int(bounding_box[4] - padding)
            end_slice = int(bounding_box[5] + padding)

            # matshow3d(
            #     title=f'{patient}',
            #     volume=sequence,
            #     every_n=15,  # Show every 6th slice
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
                d[key] = sequence
                return d

            # Check for cropping
            if self.min_size[2] < (end_slice - start_slice):
                print(f'CROPPING: {(end_slice - start_slice)} to {self.min_size[2]}')
            d[key] = cropped

            # matshow3d(
            #     title=f'{patient} Cropped: {[end_row -start_row, end_col - start_col, end_slice - start_slice]}',
            #     volume=cropped,
            #     every_n=1,  # Show every n slice
            #     cmap='gray',
            #     figsize=(10, 10)
            # )
            # plt.show()

        return d




# class CropROId(MapTransform):
#     def __init__(self, keys: KeysCollection, bbox_key='Bounding Box', margin=10, min_size=(128, 128, 50)):
#         super().__init__(keys)
#         self.bbox_key = bbox_key
#         self.margin = margin
#         self.min_size = min_size
#
#     def __call__(self, data):
#         d = dict(data)
#         patient = d.get('Patient ID')
#         bounding_box = d.get(self.bbox_key)
#
#         if bounding_box is None:
#             print(f"Warning: No bbox found for patient {patient}")
#             return d
#
#         for key in self.keys:
#             sequence = d[key]
#             C, H, W, D = sequence.shape
#
#             # Extract bbox
#             start_row, end_row = bounding_box[0], bounding_box[1]
#             start_col, end_col = bounding_box[2], bounding_box[3]
#             start_slice, end_slice = bounding_box[4], bounding_box[5]
#
#             # CLAMP to valid bounds first
#             start_row = max(0, min(start_row, H - 1))
#             end_row = max(start_row + 1, min(end_row, H))
#             start_col = max(0, min(start_col, W - 1))
#             end_col = max(start_col + 1, min(end_col, W))
#             start_slice = max(0, min(start_slice, D - 1))
#             end_slice = max(start_slice + 1, min(end_slice, D))
#
#             # Calculate current sizes
#             h_size = end_row - start_row
#             w_size = end_col - start_col
#             d_size = end_slice - start_slice
#
#             # Expand to minimum size (centered expansion with margin)
#             def expand_dimension(start, end, current_size, min_size, max_bound, margin):
#                 target_size = max(min_size, current_size) + 2 * margin
#                 needed = target_size - (end - start)
#
#                 if needed > 0:
#                     expand_before = needed // 2
#                     expand_after = needed - expand_before
#
#                     new_start = max(0, start - expand_before)
#                     new_end = min(max_bound, end + expand_after)
#
#                     # Adjust if hit boundary
#                     if new_start == 0:
#                         new_end = min(max_bound, target_size)
#                     elif new_end == max_bound:
#                         new_start = max(0, max_bound - target_size)
#
#                     return new_start, new_end
#                 return start, end
#
#             start_row, end_row = expand_dimension(start_row, end_row, h_size, self.min_size[0], H, self.margin)
#             start_col, end_col = expand_dimension(start_col, end_col, w_size, self.min_size[1], W, self.margin)
#             start_slice, end_slice = expand_dimension(start_slice, end_slice, d_size, self.min_size[2], D, self.margin)
#
#             # Crop
#             cropped = sequence[:, start_row:end_row, start_col:end_col, start_slice:end_slice]
#
#             # Safety check
#             if 0 in cropped.shape:
#                 print(f'ERROR: Zero dimension for {patient}, using full image')
#                 d[key] = sequence
#             else:
#                 d[key] = cropped
#
#         return d