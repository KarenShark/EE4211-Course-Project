import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from grad_cam import GradCAM
from ws_utils import get_data, load_vgg

def generate_vgg_cam_masks(OUTPUT_DIR, MODEL_PATH, device='cpu', method='gradcam', data_percentage=1.0):
    """
    Generate cams using a pretrained VGG model and save the results:
        - Loads a VGG model from the specified path.
        - Applies Grad-CAM to generate heatmaps for each image.
        - Saves the input image tensors and corresponding CAM masks to disk.
    Args:
        OUTPUT_DIR (str): Directory to save output images and CAM masks.
        MODEL_PATH (str): Path to the pretrained VGG model weights.
        device (str): Device to run model inference on ('cpu' or 'cuda').
        method (str): CAM method to use (currently only supports 'gradcam').
        data_percentage (float): Percentage of data to use (0-1). Default: 1.0 (full dataset).
    """

    #device
    print(f"Using device: {device}")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    num_classes = 37
    vgg = load_vgg(model_path=MODEL_PATH, device=device, num_classes=num_classes)

    selected_cam = GradCAM(model=vgg, target_layer=vgg.features[-1])

    images_dir = os.path.join(OUTPUT_DIR, 'images')
    masks_dir = os.path.join(OUTPUT_DIR, 'masks')
    os.makedirs(images_dir)
    os.makedirs(masks_dir)

    # CRITICAL: Use explicit parameters to ensure reproducible data split
    train_loader, _, _ = get_data(
        batch_size=1, 
        data_percentage=data_percentage,
        target_types='category',  # For classification (CAM generation needs class labels)
        use_validation=True,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42  # Fixed seed for reproducibility
    )
    idx = 0
    
    total_images = len(train_loader)
    print(f"\n📊 Generating CAM masks for {total_images} images...")
    print(f"   Using device: {device}")
    print(f"   This may take a while...\n")

    #generate cams here
    for images, _ in tqdm(train_loader, desc="Generating CAM masks", unit="img"):
        # Move images to the correct device (GPU/CPU)
        images = images.to(device)
        cams = selected_cam.generate_cam(input_images=images)
        assert len(images) == len(cams)
        for i, cam in enumerate(cams):
            image = images[i]
            image_filename = f"image_{idx}.pt"
            torch.save(image, os.path.join(images_dir, image_filename))
            cam_filename = f'mask_{idx}.npy'
            np.save(os.path.join(masks_dir, cam_filename), cam)
            idx += 1
    
    print(f"\n✅ CAM generation completed!")
    print(f"   Total masks generated: {idx}")
    print(f"   Saved to: {OUTPUT_DIR}\n")

