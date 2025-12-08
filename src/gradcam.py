from monai.visualize import GradCAM
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomGradCAM:
    def __init__(self, model, loader, num_heatmaps, batch_size, target_patient):
        self.model = model
        self.loader = loader
        self.num_heatmaps = num_heatmaps
        self.batch_size = batch_size
        # DataLoader Iterator
        self.loader_iter = iter(self.loader)
        self.current_batch = None
        self.batch_index = 0
        self.target_patient = target_patient

    def get_next_image(self):
        # Keep iterating until we find target patient
        while True:
            # If we've exhausted current batch or don't have one, get next batch
            if self.current_batch is None or self.batch_index >= len(self.current_batch["Window"]):
                try:
                    self.current_batch = next(self.loader_iter)
                    self.batch_index = 0
                except StopIteration:
                    # Reset iterator if we've gone through all data
                    self.loader_iter = iter(self.loader)
                    self.current_batch = next(self.loader_iter)
                    self.batch_index = 0

            # Get DataLoader data
            images = self.current_batch["Window"]
            features = self.current_batch["Features"]
            labels = self.current_batch["Label"]
            patients = self.current_batch["Patient ID"]

            # Get current image and increment index
            image = images[self.batch_index].to(device)
            feature = features[self.batch_index].to(device)
            label = labels[self.batch_index].to(device)
            patient = patients[self.batch_index]
            self.batch_index += 1

            # If target_patient is specified, filter for that patient
            if self.target_patient is not None:
                if patient == self.target_patient:
                    return image, feature, label, patient
                # Otherwise continue to next image
            else:
                return image, feature, label, patient

    def get_heatmap(self):
        image, features, label, patient = self.get_next_image()
        image = image.unsqueeze(0)
        print(f'Original Image Shape: {image.shape}')
        class ModelWrapper(nn.Module):
            def __init__(self, model, feature, batch_size):
                super().__init__()
                self.model = model
                self.features = feature
                self.batch_size = batch_size

            def __call__(self, gradcam_image):
                # Needs to be of shape: [self.n_batches, 3, Height, Width]
                print(f'Gradcam Image Shape: {gradcam_image.shape}')
                # print(f'Occlusion Image Shape After Unsqueeze: {occlusion_image.shape}')
                # Features have batch size of 1
                features_batch = self.features.expand(self.batch_size, -1)  # Expands features to batch size of n_batches
                print(f'Batch Features Shape: {features_batch.shape}')
                prediction_T = self.model(gradcam_image, features_batch)
                return prediction_T

        wrapped_model = ModelWrapper(self.model, features, self.batch_size)
        target_layer = "model.image_encoder.backbone.layer4.0.conv2"
        cam = GradCAM(
            nn_module=wrapped_model,
            target_layers=target_layer
        )
        heatmap = cam(x=image, class_idx=None)
        # Convert to numpy and detach from gpu
        image_np = image[0].detach().cpu().numpy()  # [3, H, W]
        heatmap_np = heatmap[0, 0].detach().cpu().numpy()  # [H, W]
        label_np = label[0].detach().cpu().numpy()
        return image_np, heatmap_np, label_np, patient

    def plot_heatmap(self):
        image, heatmap, label, patient = self.get_heatmap()
        # Use first channel as grayscale image
        base_img = image[0]  # or np.mean(image_np, axis=0)
        # Normalize heatmap to [0,1]
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Overlay
        plt.figure(figsize=(12, 4))
        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(base_img, cmap='gray')
        plt.title(f'Original Image Prediction: {label}')
        plt.axis('off')
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_norm, cmap='jet')
        plt.title(f'GradCAM Heatmap, Patient: {patient}')
        plt.axis('off')
        # Heatmap Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(base_img, cmap='gray')
        plt.imshow(heatmap_norm, cmap='jet', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.show()