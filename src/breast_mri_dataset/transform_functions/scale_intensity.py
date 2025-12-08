from monai.transforms import MapTransform, ScaleIntensityRange
from monai.config import KeysCollection

# Custom Min-Max Normalization
class ScaleIntensity(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
    ):
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            sequence = d[key]
            cropper = ScaleIntensityRange(
                a_min=float(sequence.min().item()),
                a_max=float(sequence.max().item()),
                # a_max=1500.0,
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
            d[key] = cropper(d[key])
        return d