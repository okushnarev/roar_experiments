# modify_dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import os
from saliency import get_saliency_map  # Import from the file above
import torchvision
import matplotlib.pyplot as plt
import copy
class ModifiedCIFAR10Dataset(Dataset):
    """
    A PyTorch Dataset that wraps an original CIFAR-10 dataset (train or test),
    calculates saliency maps for a given model and method, removes a percentage
    of the most important pixels, and applies standard normalization.
    """

    def __init__(self, original_dataset, model, saliency_method_name,
                 removal_percentage, dataset_mean, dataset_std, device='cpu'):
        """
        Args:
            original_dataset (Dataset): The original CIFAR-10 train or test dataset.
                                        Expected to return (image, label) where image
                                        is a PIL image or tensor before normalization.
            model (nn.Module): The trained model to generate saliency maps.
            saliency_method_name (str): Name of the saliency method ('random', 'grad', 'sg_sq_grad').
            removal_percentage (float): Percentage of pixels to remove (e.g., 0.2, 0.5, 0.9).
            dataset_mean (torch.Tensor or list/tuple): Per-channel mean of the original dataset (range [0,1]). Shape (3,).
            dataset_std (torch.Tensor or list/tuple): Per-channel std dev of the original dataset. Shape (3,).
            device (str): Device ('cpu' or 'cuda') to run model inference and saliency calculation.
        """
        self.original_dataset = original_dataset
        self.model = model.to(device)
        self.model.eval()  # Ensure model is in eval mode
        self.saliency_method_name = saliency_method_name
        self.removal_percentage = removal_percentage
        self.device = device

        # Store mean and std for normalization/denormalization
        # Ensure they are tensors on the correct device and have shape (3, 1, 1) for broadcasting
        self.norm_mean = torch.tensor(dataset_mean, dtype=torch.float32).view(3, 1, 1).to(device)
        self.norm_std = torch.tensor(dataset_std, dtype=torch.float32).view(3, 1, 1).to(device)

        # The replacement value is the mean of the *unnormalized* dataset [0,1]
        self.replacement_value = torch.tensor(dataset_mean, dtype=torch.float32).view(3, 1,
                                                                                      1)  # Keep on CPU until needed

        # Standard normalization transform to apply *after* modification
        self.normalize_transform = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        # Transform to get tensor in [0,1] range from original dataset (if it returns PIL)
        self.to_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 1. Get original image and label
        original_image, label = self.original_dataset[idx]

        # Convert PIL image to Tensor [0, 1] if necessary
        if not isinstance(original_image, torch.Tensor):
            img_tensor_01 = self.to_tensor_transform(original_image)  # Shape: (3, H, W), Range [0, 1]
        else:
            # Assume tensor is already in [0,1] if it came directly from dataset without ToTensor
            img_tensor_01 = original_image.clone()

        # Ensure tensor is on the correct device
        img_tensor_01 = img_tensor_01.to(self.device)

        # 2. Prepare image for saliency map generation (normalize + add batch dim)
        normalized_img = (img_tensor_01 - self.norm_mean) / self.norm_std
        input_batch = normalized_img.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)

        # 3. Generate saliency map
        # --- Start GRADIENT CALCULATION BLOCK (if needed) ---
        if self.saliency_method_name in ['grad', 'sg_sq_grad']:
            # Enable gradient calculation for these methods
            saliency_map = get_saliency_map(self.saliency_method_name, self.model, input_batch)
        else:
            # For 'random' or other methods not needing grads, use no_grad context
            with torch.no_grad():
                saliency_map = get_saliency_map(self.saliency_method_name, self.model, input_batch)
        # --- End GRADIENT CALCULATION BLOCK ---

        # Saliency map shape: (1, H, W) -> remove batch dim -> (H, W)
        saliency_map = saliency_map.squeeze(0).cpu()  # Move to CPU for numpy/mask operations

        # 4. Determine pixels to remove
        height, width = saliency_map.shape
        num_pixels_total = height * width
        num_pixels_to_remove = int(np.floor(self.removal_percentage * num_pixels_total))

        if num_pixels_to_remove <= 0:
            # No pixels to remove, just normalize the original [0,1] tensor
            modified_img_01 = img_tensor_01.cpu()  # Ensure it's on CPU before normalization
            normalized_modified_img = self.normalize_transform(modified_img_01)
            return normalized_modified_img, label

        if num_pixels_to_remove >= num_pixels_total:
            # Remove all pixels, replace with mean
            modified_img_01 = self.replacement_value.repeat(1, height, width)  # Shape: (3, H, W)
            # Ensure it's on CPU before normalization if replacement value was on CPU
            normalized_modified_img = self.normalize_transform(modified_img_01.cpu())
            return normalized_modified_img, label

        # Find the indices of the top 'k' pixels to remove
        # Flatten the saliency map to rank pixels easily
        flat_saliency = saliency_map.flatten()  # Shape: (H * W,)

        # Check for NaNs or Infs in saliency map which can cause issues with topk
        if torch.isnan(flat_saliency).any() or torch.isinf(flat_saliency).any():
            print(
                f"Warning: NaN/Inf detected in saliency map for index {idx}, method {self.saliency_method_name}. Using random mask.")
            # Fallback to a random mask in case of bad saliency values
            top_indices = torch.randperm(num_pixels_total)[:num_pixels_to_remove]
        else:
            # Use argpartition for efficiency: find k-th largest value, partition around it
            # We want the indices of the *largest* values
            # threshold_value = torch.kthvalue(-flat_saliency, num_pixels_to_remove).values # Using negative to find smallest of negatives (largest originals)
            # mask_flat = -flat_saliency <= threshold_value
            # Alternative: Get indices of topk
            _, top_indices = torch.topk(flat_saliency, k=num_pixels_to_remove)

        # Create a mask (True where pixels should be removed)
        mask_flat = torch.zeros_like(flat_saliency, dtype=torch.bool)
        mask_flat[top_indices] = True
        remove_mask = mask_flat.view(height, width)  # Shape: (H, W)

        # 5. Create the modified image
        modified_img_01 = img_tensor_01.cpu()  # Work on CPU copy
        replacement_value_cpu = self.replacement_value  # Already on CPU

        # Apply mask for each channel
        for c in range(modified_img_01.shape[0]):  # Iterate through channels (3)
            modified_img_01[c][remove_mask] = replacement_value_cpu[c, 0, 0]  # Assign mean value for this channel

        # 6. Apply standard normalization to the modified image
        normalized_modified_img = self.normalize_transform(modified_img_01)

        return normalized_modified_img, label


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    print("Testing ModifiedCIFAR10Dataset...")
    # --- Dummy Model and Data ---
    from model import SimpleCNN  # Assumes model.py is in the same directory

    dummy_model = SimpleCNN()
    # Load CIFAR-10 once to get an instance
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]
    # Load raw dataset (PIL images)
    raw_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    # --- Create Modified Datasets (Examples) ---
    print("Creating example modified dataset (GRAD, 50%)...")
    # Need to load the *actual* trained model for GRAD/SG_SQ methods to work meaningfully
    # For testing the script structure, we can proceed with the dummy, but gradients will be nonsensical.
    # Let's load the baseline model if it exists
    baseline_model_path = './cifar_baseline.pth'
    if os.path.exists(baseline_model_path):
        print(f"Loading baseline model from {baseline_model_path}")
        dummy_model.load_state_dict(torch.load(baseline_model_path, map_location='cpu'))  # Load to CPU for testing
    else:
        print("Warning: Baseline model not found. Using untrained model for testing.")

    modified_grad_50 = ModifiedCIFAR10Dataset(
        original_dataset=raw_trainset,
        model=dummy_model,  # Use trained model here
        saliency_method_name='sg_sq_grad',
        removal_percentage=0.50,
        dataset_mean=cifar_mean,
        dataset_std=cifar_std,
        device='cpu'  # Use CPU for testing consistency
    )

    print("Creating example modified dataset (Random, 50%)...")
    modified_random_50 = ModifiedCIFAR10Dataset(
        original_dataset=raw_trainset,
        model=dummy_model,  # Model not used for 'random'
        saliency_method_name='random',
        removal_percentage=0.50,
        dataset_mean=cifar_mean,
        dataset_std=cifar_std,
        device='cpu'
    )

    print(modified_grad_50)
    # --- Test loading a sample ---
    print(f"Length of original dataset: {len(raw_trainset)}")
    print(f"Length of modified GRAD dataset: {len(modified_grad_50)}")

    print("\nLoading sample 0 from modified GRAD 50% dataset:")
    mod_img, mod_label = modified_grad_50[0]
    print(f"  Label: {mod_label}")
    print(f"  Image Tensor Shape: {mod_img.shape}")
    print(f"  Image Tensor Type: {mod_img.dtype}")
    # Check min/max values - should be roughly centered around 0 due to normalization
    print(f"  Image Tensor Min: {mod_img.min():.4f}, Max: {mod_img.max():.4f}, Mean: {mod_img.mean():.4f}")

    print("\nLoading sample 1 from modified Random 50% dataset:")
    mod_img_rand, mod_label_rand = modified_random_50[0]
    print(f"  Label: {mod_label_rand}")
    print(f"  Image Tensor Shape: {mod_img_rand.shape}")
    # Check min/max values - likely heavily influenced by the mean value due to 90% replacement
    print(
        f"  Image Tensor Min: {mod_img_rand.min():.4f}, Max: {mod_img_rand.max():.4f}, Mean: {mod_img_rand.mean():.4f}")

    # --- Visualize (Optional - requires matplotlib) ---

    # Small change to handle potential tensor device
    def imshow(img_tensor, mean, std, ax, title=""):
        """Display helper with unnormalization"""
        img_tensor = img_tensor.cpu()
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img_tensor * std + mean
        npimg = img.clamp(0, 1).numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title, pad=10)  # Title padding
        ax.axis('off')


    # Create 1x3 figure (single row, three columns)
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))  # Width >> Height

    # --- Original Image ---
    axs[0].imshow(raw_trainset[0][0])
    axs[0].set_title("Original Image", pad=10)
    axs[0].axis('off')

    # --- Modified GRAD 50% ---
    imshow(mod_img, cifar_mean, cifar_std, axs[1], "Modified GRAD 50%")

    # --- Modified Random 50% ---
    imshow(mod_img_rand, cifar_mean, cifar_std, axs[2], "Modified Random 50%")

    # Final touch: Adjust spacing
    plt.tight_layout()  # Extra horizontal padding
    plt.show()


