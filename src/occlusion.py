from monai.visualize import OcclusionSensitivity
import torch
import cv2
import matplotlib.pyplot as plt
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# https://github.com/Project-MONAI/tutorials/blob/main/3d_classification/densenet_training_array.ipynb
class Occlusion_Sensitivity:
    def __init__(self, model, loader):
        # "Mask Size: Size of box to be occluded, centred on the central voxel.
        # If a single number is given, this is used for all dimensions. If a sequence
        # is given, this is used for each dimension individually".
        self.mask_size = 12
        self.n_batches = 10  #  – Number of images in a batch for inference.
        self.model = model
        self.loader = loader

    def get_next_image(self):
        batch = next(iter(self.loader))
        images = batch["image_paths"]
        features = batch["features"]  # shape: [batch_size, num_features]
        labels = batch["label"]  # shape: [batch_size, num_labels]

        # Only want 1 volume image
        return images[0].to(device), features[0].to(device), labels[0].to(device)


    def get_heatmap(self):
        image, features, label = self.get_next_image()
        image = image.unsqueeze(0)
        print(f'Original Image Shape: {image.shape}')
        class ModelWrapper:
            def __init__(self, model, feature, batch_size):
                self.model = model
                self.features = feature
                self.batch_size = batch_size

            def __call__(self, occlusion_image):
                # Needs to be of shape: [1, self.n_batches, 256, 256, 160]
                print(f'Occlusion Image Shape: {occlusion_image.shape}')
                # occlusion_image = occlusion_image.unsqueeze(0)
                # print(f'Occlusion Image Shape After Unsqueeze: {occlusion_image.shape}')
                # Features have batch size of 1
                features_batch = self.features.expand(self.batch_size, -1)  # Expands features to batch size of n_batches
                print(f'Batch Features Shape: {features_batch.shape}')
                prediction_T, _, _ = self.model(occlusion_image, features_batch)
                return prediction_T

        wrapped_model = ModelWrapper(self.model, features, self.n_batches)
        occlusion_sensitivity = OcclusionSensitivity(
            nn_module=wrapped_model,
            mask_size=self.mask_size,
            n_batch=self.n_batches,
            verbose=True)
        # Only get a single slice to save time.
        # For the other dimensions (channel, width, height), use
        # -1 to use 0 and img.shape[x]-1 for min and max, respectively
        print(f'Image Shape: {image.shape}')
        depth_slice = image.shape[2] // 2
        print(f'Depth Slice: {depth_slice}')
        occlusion_b_box = [depth_slice - 1, depth_slice, -1, -1, -1, -1]
        print(f'Occlusion Box: {occlusion_b_box}')
        # Results
        print(image.dtype)
        occlusion_result, _ = occlusion_sensitivity(x=image, b_box=occlusion_b_box)
        print(f'Occlusion Result1: {occlusion_result}')
        occlusion_result = occlusion_result[0, label.argmax().item()][None]
        print(f'Occlusion Result2: {occlusion_result}')
        return image, occlusion_result

    def plot_heatmap(self, alpha=0.4):
        image, heatmap = self.get_heatmap()
        print(f'Original Image Shape : {image.shape}')
        image = image.detach().cpu()
        # Color Maps
        # heatmap_hot = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        # heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Image Heatmap Overlay
        # image_overlay_hot = cv2.addWeighted(image, alpha, heatmap_hot, 1 - alpha, 0)
        # image_overlay_jet = cv2.addWeighted(image, alpha, heatmap_jet, 1 - alpha, 0)

        # fig, axes = plt.subplots(1, 2, figsize=(25, 15), facecolor="white")
        #
        # for i, im in enumerate([img[:, :, depth_slice, ...], occ_result]):
        #     cmap = "gray" if i == 0 else "jet"
        #     ax = axes[i]
        #     im_show = ax.imshow(np.squeeze(im[0][0].detach().cpu()), cmap=cmap)
        #     ax.axis("off")
        #     fig.colorbar(im_show, ax=ax)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.xlabel('Original Slice')

        plt.subplot(1, 3, 2)
        # plt.imshow(image_overlay_hot)
        plt.xlabel('')

        plt.subplot(1, 3, 3)
        # plt.imshow(image_overlay_jet)
        plt.xlabel('')
        return heatmap