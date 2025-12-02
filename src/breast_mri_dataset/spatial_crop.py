from monai.transforms import MapTransform, SpatialCrop
from monai.config import KeysCollection
import numpy as np

class SpatialCropBBoxd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            bbox_key: str = "Bounding Box",
            target_size: tuple[int, int, int] = (128, 128, 80),
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.bbox_key = bbox_key
        self.target_size = np.array(target_size)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        # Get current image spatial shape (from first key)
        cropped_image = d[self.keys[0]]
        H, W, D = cropped_image.shape[1:]
        roi_center = [H // 2, W // 2, D // 2]
        # Apply the same crop to all keys
        cropper = SpatialCrop(roi_size=self.target_size, roi_center=roi_center)
        for key in self.keys:
            if key in d:
                d[key] = cropper(d[key])
        return d