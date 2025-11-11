from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import torch

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset class for image segmentation tasks.
    Args:
        images (list): List of file paths to input images.
        masks (list): List of file paths to corresponding segmentation masks.
        transforms (callable or tuple, optional): Transform(s) to apply to both images and masks.
    """

    def __init__(self, images: list, masks: list, transforms=None) -> None:
        """
        Initialize the dataset with image and mask paths. 
        """
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        Returns:
            int: Number of image-mask pairs.
        """
        return len(self.images)

    def __getitem__(self, index: int):
        """
        Load and return a transformed image and its corresponding processed mask.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, mask) where image is a 3D tensor and mask is a float tensor
                suitable for binary segmentation.
        """
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') #expect 1, 2, 3

        if self.transforms is not None:
            if isinstance(self.transforms, tuple):
                image_transform, mask_transform = self.transforms
                image = image_transform(image)
                mask = mask_transform(mask)
            else:
                image = self.transforms(image)
                mask = self.transforms(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        mask = mask * 255
        mask = mask.squeeze().to(torch.int64)
        mask = mask - 1
        #merge unknown region with foreground 
        mask = torch.where(mask == 2, torch.tensor(1, dtype=torch.int64, device=mask.device), mask)
        if mask.ndimension() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        
        return image, mask

def get_binary_and_valid_mask(trimap_path):
    """
    Convert a trimap image into a binary mask and a valid region mask.
    Args:
        trimap_path (str): Path to the grayscale trimap image.
    Returns:
        tuple: (binary_mask, valid_mask)
            - binary_mask (np.ndarray): 2D binary mask where background=0 and foreground=1.
            - valid_mask (np.ndarray): Boolean mask indicating which pixels are valid (not unknown).
    """

    trimap_img = Image.open(trimap_path).convert("L")
    trimap_array = np.array(trimap_img).astype(np.uint8)
    
    if trimap_array.max() > 10:
        remap = {0: 1, 128: 2, 255: 3}
        trimap_array = np.vectorize(remap.get)(trimap_array)
    
    valid_mask = trimap_array != 3
    
    #convert to binary mask here
    binary_mask = trimap_array - 1
    #merge unknown region with foreground 
    binary_mask[binary_mask == 2] = 1
    
    return binary_mask, valid_mask

def save_binary_masks(input_folder, output_folder):
    """
    Convert all PNG trimap images in a folder into binary masks and save them.
    Args:
        input_folder (str): Directory containing trimap PNG images.
        output_folder (str): Directory to save the generated binary masks.
    """

    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png') and not filename.startswith("._"):
            input_path = os.path.join(input_folder, filename)
            try:
                binary_mask, valid_mask = get_binary_and_valid_mask(input_path)
            except Exception as e:
                continue
            
            binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
            
            output_path = os.path.join(output_folder, filename)
            binary_img.save(output_path)