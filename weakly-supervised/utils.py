import os
import random

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

VGG_NORM_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])  
])


def set_random_seed(seed):
    """
    Set random seed to ensure reproducibility.
    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)  # Set the seed for PyTorch
    random.seed(seed)  # Set the seed for Python's random module
    np.random.seed(seed)  # Set the seed for NumPy


def get_data(batch_size, data_percentage=1.0, train_percentage=None, test_percentage=None, 
             transform=VGG_NORM_TRANSFORM, target_types='category', target_transform=None, 
             use_validation=True, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Load and return DataLoaders for the Oxford-IIIT Pet dataset with train/val/test split.
    
    Now merges trainval and test datasets, then re-splits into 70/15/15 for standard CV practice.
    
    Args:
        batch_size (int): Number of samples per batch.
        data_percentage (float): Percentage of total data to use (0-1). Default: 1.0 (full dataset).
        train_percentage (float): DEPRECATED - use data_percentage instead.
        test_percentage (float): DEPRECATED - use data_percentage instead.
        transform (callable): Transform to apply to input images.
        target_types (str): Type of target to return ('category' or 'segmentation').
        target_transform (callable): Optional transform for targets.
        use_validation (bool): Whether to split trainval into train and validation sets.
        train_ratio (float): Ratio of train data from all data (default: 0.7).
        val_ratio (float): Ratio of validation data from all data (default: 0.15).
        seed (int): Random seed for splitting.
    Returns:
        tuple: (train_loader, val_loader, test_loader) if use_validation=True, 
               else (train_loader, test_loader) for backward compatibility.
    """
    
    # Handle deprecated parameters
    if train_percentage is not None or test_percentage is not None:
        print("⚠️  Warning: train_percentage and test_percentage are deprecated. Use data_percentage instead.")
        if train_percentage is not None and train_percentage < 1.0:
            data_percentage = train_percentage

    if target_types == 'segmentation':
        target_transform = lambda x: np.array(x.convert("L").resize((224, 224), Image.NEAREST))
    
    # Load full datasets
    trainval_dataset = OxfordIIITPet(root='data', split='trainval', target_types=target_types, transform=transform,
                                     download=True, target_transform=target_transform)
    test_dataset = OxfordIIITPet(root='data', split='test', target_types=target_types, transform=transform,
                                 target_transform=target_transform)
    
    print(f"\nOriginal Oxford-IIIT Pet dataset:")
    print(f"  Trainval: {len(trainval_dataset)}")
    print(f"  Test:     {len(test_dataset)}")
    
    if use_validation:
        # Merge all data and re-split into 70/15/15
        from torch.utils.data import ConcatDataset
        all_dataset = ConcatDataset([trainval_dataset, test_dataset])
        total_size = len(all_dataset)
        
        print(f"  Total:    {total_size}")
        print(f"\nMerging all data and re-splitting into {train_ratio*100:.0f}/{val_ratio*100:.0f}/{(1-train_ratio-val_ratio)*100:.0f}...")
        
        # Create indices for all data
        all_indices = list(range(total_size))
        
        # Set seed for reproducible split
        random.seed(seed)
        random.shuffle(all_indices)
        
        # Calculate split sizes (70/15/15 of total data)
        test_ratio = 1.0 - train_ratio - val_ratio
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size + val_size]
        test_indices = all_indices[train_size + val_size:]
        
        # Apply data_percentage if specified (for quick testing)
        if data_percentage < 1.0:
            keep_train = int(len(train_indices) * data_percentage)
            keep_val = int(len(val_indices) * data_percentage)
            keep_test = int(len(test_indices) * data_percentage)
            train_indices = train_indices[:keep_train]
            val_indices = val_indices[:keep_val]
            test_indices = test_indices[:keep_test]
        
        # Create subsets
        train_dataset = Subset(all_dataset, train_indices)
        val_dataset = Subset(all_dataset, val_indices)
        test_dataset_new = Subset(all_dataset, test_indices)
        
        print(f"\nNew train/val/test split (seed={seed}):")
        print(f"  Train: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
        print(f"  Val:   {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
        print(f"  Test:  {len(test_dataset_new)} ({len(test_dataset_new)/total_size*100:.1f}%)")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    else:
        # Original behavior for backward compatibility
        train_dataset = trainval_dataset
        
        print(f"Taking {train_percentage * 100}% of train dataset and {test_percentage * 100}% of test dataset")
        if train_percentage < 1.0:
            train_size = int(train_percentage * len(train_dataset))
            train_dataset = Subset(train_dataset, random.sample(range(len(train_dataset)), train_size))
        if test_percentage < 1.0:
            test_size = int(test_percentage * len(test_dataset))
            test_dataset = Subset(test_dataset, random.sample(range(len(test_dataset)), test_size))
        
        print(f"Final Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

def overlay_cam(input_image, cam, alpha=0.5):
    """
    Overlay a CAM heatmap on top of the original image.
    Args:
        input_image (Tensor): The input image tensor.
        cam (np.ndarray): The CAM heatmap.
        alpha (float): Transparency of the overlay.
    Returns:
        tuple: (original image as np.ndarray, overlay image with CAM)
    """

    # Convert image to numpy
    input_image = input_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    input_image = np.clip(input_image, 0, 1)

    # Apply a colormap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay the heatmap onto the image
    overlayed_image = (heatmap * alpha + input_image * 255 * (1 - alpha)) / 255.0

    return input_image, overlayed_image


def load_vgg(model_path, device, num_classes):
    """
    Load a VGG16 model from saved weights and modify it for a specific number of output classes.
    Args:
        model_path (str): Path to the saved model file.
        device (str or torch.device): Device to load the model onto.
        num_classes (int): Number of output classes.
    Returns:
        torch.nn.Module: The loaded VGG model.
    """
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def visualize_prediction(model, image_tensor, foreground_threshold=0.5, target_mask=None, save_img_path=None):
    """
    Predict and show segmentation results on a single image.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        image_tensor (Tensor): Input image tensor.
        foreground_threshold (float): Threshold to binarize the predicted mask.
        target_mask (Tensor, optional): Ground truth mask for comparison.
        save_img_path (str, optional): Path to save the visualization image.
    Returns:
        np.ndarray: Binary predicted mask as a NumPy array.
    """
    device = next(model.parameters()).device
    model.eval()

    # Ensure image tensor is on the right device
    image_tensor = image_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))['out']  # Add batch dimension if needed
        output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        # output = output.squeeze().argmax(dim=0)

    # Convert tensors to numpy for visualization
    pred_mask = output.squeeze().cpu().numpy()
    pred_mask = pred_mask > foreground_threshold
    # pred_mask = output.cpu().numpy()
    input_image = image_tensor.cpu().permute(1, 2, 0).numpy()

    # Denormalize the image if it was normalized
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = std * input_image + mean
    input_image = np.clip(input_image, 0, 1)

    # Plot results
    fig, ax = plt.subplots(1, 4 if target_mask is not None else 2, figsize=(15, 5))

    # Plot original image if provided, otherwise use the processed input
    ax[0].imshow(input_image)
    ax[0].set_title('Original Image')

    # Plot predicted mask
    ax[1].imshow(pred_mask, cmap='gray')
    ax[1].set_title('Predicted Mask')

    # Plot ground truth mask if provided
    if target_mask is not None:
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.squeeze().cpu().numpy()
        # target_mask[target_mask == 3] = 1
        # target_mask[target_mask == 2] = 0
        ax[2].imshow(target_mask, cmap='jet')
        ax[2].set_title('Ground Truth')

    # Remove axes for cleaner visualization
    for a in ax:
        a.axis('off')
    # Create overlay visualization (prediction on top of image)
    ax[3].imshow(input_image)
    ax[3].imshow(pred_mask, cmap='gray', alpha=0.4)
    ax[3].set_title('Predict Mask Overlay Image')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    if save_img_path is not None:
        plt.savefig(save_img_path)
    plt.show()

    return pred_mask


def denormalize_image(image):
    """
    Convert normalised image tensor back to a uint8 image in [0, 255] range.
    Args:
        image (Tensor): Normalized image tensor.
    Returns:
        np.ndarray: Denormalized image as a NumPy array.
    """
    denorm_image = image.cpu().permute(1, 2, 0).numpy()
    # Denormalize the image if it was normalized
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denorm_image = std * denorm_image + mean
    denorm_image = np.clip(denorm_image, 0, 1)
    return (denorm_image * 255).astype(np.uint8)


class ImageCamMaskDataset(Dataset):
    """
    A Dataset class for loading GradCAM weak segmentation data.
    Args:
        data_dir (str): Directory containing 'images/' and 'masks/' subfolders.
        percentage (float): Percentage of dataset to use.
        use_crf (bool): Whether to refine masks using CRF.
    Methods:
        __len__: Return the number of samples.
        __getitem__: Return a single image and its CAM or CRF-refined mask.
    """
    def __init__(self, data_dir, percentage=1.0, use_crf=True):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.pt')]
        self.mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith('.npy')]
        self.image_files.sort()
        self.mask_files.sort()
        assert len(self.image_files) == len(self.mask_files)
        self.dataset_size = int(percentage * len(self.image_files))
        self.image_files = self.image_files[:self.dataset_size]
        self.mask_files = self.mask_files[:self.dataset_size]
        self.use_crf = use_crf

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = torch.load(img_path)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))
        cam_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)
        if self.use_crf:
            denorm_image = denormalize_image(image)
            crf_cam = apply_crf(cam=cam_resized, image=denorm_image)
            crf_cam_norm = (crf_cam - crf_cam.min()) / (crf_cam.max() - crf_cam.min())
            return image, crf_cam_norm
        return image, torch.from_numpy(cam_resized).float()


def apply_crf(cam, image, n_classes=2):
    """
    Refine a CAM heatmap using DenseCRF. 
    Args:
        cam (np.ndarray): Initial CAM heatmap.
        image (np.ndarray): Corresponding RGB image.
        n_classes (int): Number of segmentation classes.
    Returns:
        np.ndarray: CRF-refined mask for the foreground class.
    """

    image = np.ascontiguousarray(image)
    h, w = cam.shape
    d = dcrf.DenseCRF2D(w, h, n_classes)

    # Create softmax: background vs foreground
    softmax = np.zeros((n_classes, h, w), dtype=np.float32)
    softmax[1, :, :] = cam  # foreground probability
    softmax[0, :, :] = 1.0 - cam  # background probability

    # Convert to unary potentials
    unary = unary_from_softmax(softmax)
    d.setUnaryEnergy(unary)

    # Add pairwise terms
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=45, srgb=25, rgbim=image, compat=10)

    # Run inference
    Q = d.inference(1)
    preds = np.array(Q).reshape((n_classes, h, w))
    refined_mask = preds[1]  # foreground class probability

    return refined_mask