from monai.transforms import MapTransform
import random

# Custom MONAI Transform to crop a 3D volumetric image to a specified bounding box.
class CropROId(MapTransform):
    def __init__(self, keys=['image'], window_size=3, margin=0, bbox_key='Bounding Box', deterministic=False):
        super().__init__(keys)
        self.window_size = window_size
        self.margin = margin
        self.bbox_key = bbox_key
        self.deterministic = deterministic

    def __call__(self, data):
        d = dict(data)  # Create a mutable copy of the input dictionary
        bounding_box = d.get(self.bbox_key)
        if bounding_box is None:
            # If no bbox, return original data (fallback)
            print(f"Warning: No bbox found for patient {d.get('Patient ID', 'unknown')}")
            return d
        for key in self.keys:
            sequence = d[key]
            padding = self.margin / 2
            start_slice = max(0, int(bounding_box[4] - padding))  # Clamp to avoid -indices
            end_slice = min(sequence.shape[3], int(bounding_box[5] + padding)) # Clamp to avoid out of bounds indices
            depth_crop = sequence[:, :, :, start_slice:end_slice]
            crop_depth = depth_crop.shape[3]
            # Skip if not enough slices
            if crop_depth < self.window_size:
                print(f'Crop: {crop_depth.shape}')
                print(f'Depth: {crop_depth}')
                raise Exception('Insufficient Depth')

            # # Select window position
            max_start = crop_depth - self.window_size
            if self.deterministic:
                # Always use middle window for validation/test
                window_start = max_start // 2
            else:
                # Random window for training
                window_start = random.randint(0, max_start)

            window_end = window_start + self.window_size

            # Extract window and permute
            window = depth_crop[:, :, :, window_start:window_end]
            window = window.permute(0, 3, 1, 2)  # [C, 3, H, W]

            if window.shape[0] == 1:
                window = window.squeeze(0)  # [3, H, W]
            d['Window'] = window
        return d
