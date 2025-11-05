import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pydicom
import numpy as np
import imutils

class Breast_MRI(Dataset):
    def __init__(self, sequences, labels, features):
        self.target_size = (512, 512)
        # Converts to Pillow image, resizes all images to (512, 512), normalizes range to [0, 1], and converts to torch stack
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(self.target_size),
                                             transforms.ToTensor()])
        self.corrupted_files = []
        # Load sequences
        self.sequences = sequences
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # """
        #     Returns:
        #         images: [num_slices, C, H, W] tensor
        #         features: [10] tensor
        #         label: [TNM] int
        # """

        sequence_images = []
        sequence_path = self.sequences[index]['image_paths']
        label = 0
        # label = self.labels[index]
        for image_path in sequence_path:
            image = pydicom.dcmread(image_path).pixel_array
            # Image Shape: (448, 448)
            image = self.crop_image(image, image_path)
            image = self.transform(image)
            sequence_images.append(image)
            # Image Shape: torch.Size([1, 512, 512])

        sequence = torch.stack(sequence_images)  # Shape: [num_slices, C, H, W]
        return sequence, label

    def crop_image(self, image, path):
        # Convert to range [0-255] and datatype np.uint8
        new_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Reduce noise and smooth image.
        new_image = cv2.GaussianBlur(new_image, (5, 5), 0)

        # Converts image to binary (black and white) pixels above 45 are white(255). Pixels below 45 become black(0).
        # Isolates bright regions in MRI(brain tissue).
        new_image = cv2.threshold(new_image, 28, 255, cv2.THRESH_BINARY)[1]

        # Morphological Operations to smooth and clean binary mask:
        # Helps see contours easily, removes small white noise.
        new_image = cv2.erode(new_image, None, iterations=2)
        # Expands white areas back to original size.
        new_image = cv2.dilate(new_image, None, iterations=2)
        # Send as copy so data is not lost, retrieves only outermost contours(RETR_EXTERNAL).
        contours = cv2.findContours(new_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Safety Check
        if not contours:
            print("[WARNING] No contours found!")
            print(path)
            return image

        # Get the largest contour by measuring the area. The largest contour will be the outer tissue.
        contours = max(contours, key=cv2.contourArea)
        # Bounding boxes of contour/tissue. Need left, right, top, and bottom.
        ext_left = tuple(contours[contours[:, :, 0].argmin()])[0]
        ext_right = tuple(contours[contours[:, :, 0].argmax()])[0]
        ext_top = tuple(contours[contours[:, :, 1].argmin()])[0]
        ext_bottom = tuple(contours[contours[:, :, 1].argmax()])[0]

        # Slice the image through rectangular bounding box.
        cropped_image = image[ext_top[1]: ext_bottom[1], ext_left[0]: ext_right[0]]

        # Convert cropped image to uint8 for transforms
        if cropped_image.dtype == np.uint16:
            cropped_image = cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return cropped_image


