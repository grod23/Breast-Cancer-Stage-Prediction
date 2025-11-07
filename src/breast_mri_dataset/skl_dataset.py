import pydicom
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects
import numpy as np
import cv2


class SKL_Dataset:
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.target_size = (64, 64)
        self.slice_samples = 3

    # Preprocessing pipeline for scikitlearn machine learning algorithms
    # SKLearn expects numpy arrays
    # X → shape(num_samples, num_features)
    # y → shape(num_samples, )

    # Process single image
    def process_skl_image(self, image_path):
        # Read as pydicom pixel array
        image = pydicom.dcmread(image_path).pixel_array
        # Normalize between 0 and 1 (add 1e-8 to avoid division by zero)
        image = (image - image.min()) / (image.max() - image.min())  # + 1e-8)
        # Resize to target range
        image = cv2.resize(image, self.target_size)
        return image

    # Convert sequences to flat list of images
    def preprocess_skl(self):
        images = []
        labels = []
        # SKLearn can't take 3d arrays as input
        # Sample sequence of slices as a fixed number per patient
        for index, sequence in enumerate(self.sequences):
            # Skip empty sequences
            if not sequence:
                continue
            num_slices = len(sequence)

            # Safety check for sequences with little to no slices
            if self.slice_samples >= num_slices:
                indices = list(range(num_slices))
            else:
                indices = np.linspace(0, num_slices - 1, self.slice_samples, dtype=int)

            for i in indices:
                image_path = sequence[i]
                image = self.process_skl_image(image_path)

                # Segmentation images have slices stacked into 1 DICOM image
                # Skip segmentation images for now
                if image.ndim == 3:
                    continue

                images.append(image)
                # Append label to keep label size same as image size
                labels.append(self.labels[index])

        return np.array(images, dtype=np.float32), np.array(labels)

    # SKImage for extracting relevant features of images
    def extract_features(self, images):
        all_features = []

        for image in images:
            # Per Image Features
            features = []

            # Ensure correct image format
            image = (image * 255).astype(np.uint8)

            # HOG features (Histogram of Oriented Gradient): Captures shapes/edges
            # 1. optional) global image normalisation
            # 2. computing the gradient image in x and y
            # 3. computing gradient histograms
            # 4. normalising across blocks
            # 5. flattening into a feature vector
            hog_features = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True
            )

            features.extend(hog_features)

            # 2. GLCM texture features(Gray Level Co-occurence Matrices): Strong in medical imaging
            try:
                glcm = graycomatrix(
                    image,
                    distances=[1, 2],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                # Extract texture properties
                contrast = graycoprops(glcm, 'contrast').mean()
                dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
                homogeneity = graycoprops(glcm, 'homogeneity').mean()
                energy = graycoprops(glcm, 'energy').mean()
                correlation = graycoprops(glcm, 'correlation').mean()

                features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
            except:
                features.extend([0, 0, 0, 0, 0])  # Fallback

            # 3. Edge intensity features
            edges = sobel(image)
            features.extend([
                edges.mean(),
                edges.std(),
                edges.max()
            ])

            # 4. Basic intensity statistics
            features.extend([
                image.mean(),
                image.std(),
                image.min(),
                image.max(),
                np.median(image),
                np.percentile(image, 25),
                np.percentile(image, 75)
            ])

            all_features.append(features)

        return np.array(all_features, dtype=np.float32)

    def model_prep(self):
        images, labels = self.preprocess_skl()
        print(f"Sampled {len(images)} images from {len(self.sequences)} sequences")
        X = self.extract_features(images)
        # Replace all 0.5 M labels (column 2) with 0
        labels[:, 2] = np.where(labels[:, 2] == 0.5, 0, labels[:, 2])
        return X, np.array(labels)