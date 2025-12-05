import os
import numpy as np
from PIL import Image

def get_binary_and_valid_mask(trimap_path):
    """
    Load a trimap and convert it to a binary mask and a valid pixel mask.
    - Foreground (255 or 2) is mapped to 1.
    - Background (0 or 1) is mapped to 0.
    - Unknown (128 or 3) is merged into foreground for binary, but excluded in valid_mask.
    Args:
        trimap_path (str): Path to the trimap PNG file.
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - binary_mask (H, W): Foreground = 1, Background = 0.
            - valid_mask (H, W): Boolean mask indicating valid (non-unknown) pixels.
    """
    trimap_img = Image.open(trimap_path).convert("L")
    trimap_array = np.array(trimap_img).astype(np.uint8)

    # Remap from [0,128,255] to [1,2,3] if necessary.
    if trimap_array.max() > 10:
        remap = {0: 1, 128: 2, 255: 3}
        trimap_array = np.vectorize(remap.get)(trimap_array)
    
    # Valid pixels are those that are not labeled as 3 (unknown).
    valid_mask = trimap_array != 3
    
    # Subtract 1 so that 1->0 (bg), 2->1 (fg), 3->2 (unknown).
    binary_mask = trimap_array - 1
    # Merge unknown (2) with foreground (1).
    binary_mask[binary_mask == 2] = 1
    
    return binary_mask, valid_mask

def save_binary_masks_from_pickle(image_paths, trimap_dir, output_folder):
    """
    Generate binary masks from a list of image paths and save them as PNGs.
    Args:
        image_paths (list): List of paths to input images (used to derive base filenames).
        trimap_dir (str): Directory containing trimap PNGs.
        output_folder (str): Directory to save the binary mask PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        trimap_path = os.path.join(trimap_dir, base_name + ".png")

        if not os.path.exists(trimap_path):
            continue
        
        try:
            binary_mask, valid_mask = get_binary_and_valid_mask(trimap_path)
        except Exception as e:
            continue
        
        # Invert the mask so foreground is white (255) and background is black (0).
        binary_mask = 1 - binary_mask
        
        binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        output_path = os.path.join(output_folder, base_name + ".png")
        binary_img.save(output_path)
