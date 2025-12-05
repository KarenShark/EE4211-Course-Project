#!/usr/bin/env python3
"""
Pipeline Visualization Script
Generates visualizations showing intermediate results of the WSSS pipeline:
1. Original Image
2. CAM Heatmap
3. Binary Pseudo-Mask (threshold=0.05)
4. CRF Refined Mask
5. Ground Truth Mask
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from image-level-supervision (note: hyphen in directory name)
sys.path.insert(0, os.path.join(project_root, 'image-level-supervision'))
from ws_utils import denormalize_image, apply_crf


def visualize_wsss_pipeline(
    image_tensor,
    cam_heatmap,
    ground_truth_mask,
    threshold=0.05,
    use_crf=True,
    save_path=None
):
    """
    Visualize the complete WSSS pipeline for a single image.
    
    Args:
        image_tensor: Normalized image tensor [3, H, W]
        cam_heatmap: CAM heatmap [H, W] in range [0, 1]
        ground_truth_mask: Ground truth trimap [H, W]
        threshold: Threshold for binary mask generation
        use_crf: Whether to apply CRF refinement
        save_path: Path to save the figure
    """
    # Denormalize image
    img_denorm = denormalize_image(image_tensor)
    
    # Generate binary pseudo-mask
    binary_mask = (cam_heatmap >= threshold).astype(np.float32)
    
    # Apply CRF if requested
    if use_crf:
        crf_mask = apply_crf(cam=cam_heatmap, image=img_denorm)
        crf_binary = (crf_mask >= threshold).astype(np.float32)
        n_plots = 6
    else:
        n_plots = 5
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
    
    # 1. Original Image
    axes[0].imshow(img_denorm)
    axes[0].set_title('(1) Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. CAM Heatmap
    im2 = axes[1].imshow(cam_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('(2) Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. CAM Overlay on Image
    overlay = img_denorm.copy()
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('(3) CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Binary Pseudo-Mask (before CRF)
    axes[3].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'(4) Binary Mask (τ={threshold})', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    if use_crf:
        # 5. CRF Refined Mask
        im5 = axes[4].imshow(crf_mask, cmap='jet', vmin=0, vmax=1)
        axes[4].set_title('(5) CRF Refined', fontsize=12, fontweight='bold')
        axes[4].axis('off')
        plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)
        
        # 6. Ground Truth
        # Convert trimap: 1=foreground, 2=background, 3=boundary
        gt_display = ground_truth_mask.copy().astype(np.float32)
        gt_display[gt_display == 1] = 1.0  # Foreground -> white
        gt_display[gt_display == 2] = 0.0  # Background -> black
        gt_display[gt_display == 3] = 0.5  # Boundary -> gray
        axes[5].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
        axes[5].set_title('(6) Ground Truth', fontsize=12, fontweight='bold')
        axes[5].axis('off')
    else:
        # 5. Ground Truth
        gt_display = ground_truth_mask.copy().astype(np.float32)
        gt_display[gt_display == 1] = 1.0
        gt_display[gt_display == 2] = 0.0
        gt_display[gt_display == 3] = 0.5
        axes[4].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
        axes[4].set_title('(5) Ground Truth', fontsize=12, fontweight='bold')
        axes[4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()
    
    return fig


def visualize_comparison(
    image_tensor,
    cam_heatmap,
    ground_truth_mask,
    pred_mask_no_crf,
    pred_mask_crf,
    save_path=None
):
    """
    Visualize comparison of results with/without CRF.
    
    Args:
        image_tensor: Normalized image tensor
        cam_heatmap: CAM heatmap
        ground_truth_mask: Ground truth trimap
        pred_mask_no_crf: Predicted mask without CRF
        pred_mask_crf: Predicted mask with CRF
        save_path: Path to save the figure
    """
    img_denorm = denormalize_image(image_tensor)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Input and Ground Truth
    axes[0, 0].imshow(img_denorm)
    axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    im01 = axes[0, 1].imshow(cam_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    gt_display = ground_truth_mask.copy().astype(np.float32)
    gt_display[gt_display == 1] = 1.0
    gt_display[gt_display == 2] = 0.0
    gt_display[gt_display == 3] = 0.5
    axes[0, 2].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    axes[1, 0].imshow(pred_mask_no_crf, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction (w/o CRF)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_mask_crf, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction (w/ CRF)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Difference map
    diff = np.abs(pred_mask_crf - pred_mask_no_crf)
    im12 = axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('CRF Difference', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Generate sample visualizations from saved CAM masks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize WSSS pipeline intermediate results')
    parser.add_argument('--cam_dir', type=str, default='image-level-supervision/vgg_cam_masks',
                        help='Directory containing CAM masks')
    parser.add_argument('--output_dir', type=str, default='output/pipeline_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample images to visualize')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Threshold for binary mask generation')
    parser.add_argument('--use_crf', action='store_true',
                        help='Apply CRF refinement')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CAM directory exists
    if not os.path.exists(args.cam_dir):
        print(f"❌ CAM directory not found: {args.cam_dir}")
        print("Please run the image-level-supervision training first to generate CAM masks.")
        return
    
    # Load CAM masks
    images_dir = os.path.join(args.cam_dir, 'images')
    masks_dir = os.path.join(args.cam_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"❌ Images or masks subdirectory not found in {args.cam_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.pt')])[:args.num_samples]
    
    print(f"\n{'='*80}")
    print(f"Generating WSSS Pipeline Visualizations")
    print(f"{'='*80}\n")
    print(f"CAM directory: {args.cam_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Use CRF: {args.use_crf}")
    print(f"Number of samples: {len(image_files)}\n")
    
    # Load ground truth data using the EXACT same logic as CAM generation
    print("Loading ground truth data with same split as CAM generation...")
    
    # Add image-level-supervision directory to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ws_path = os.path.join(project_root, 'image-level-supervision')
    if ws_path not in sys.path:
        sys.path.insert(0, ws_path)
    
    from ws_utils import get_data, set_random_seed
    from torchvision.datasets import OxfordIIITPet
    from torch.utils.data import ConcatDataset, Subset
    from PIL import Image
    import numpy as np
    import random
    
    # CRITICAL: Set seed BEFORE any data loading to match gen_cam_masks.py
    set_random_seed(42)
    
    # Load segmentation ground truth with IDENTICAL parameters as CAM generation
    # This ensures the EXACT same data split and order
    trainval_ds_seg = OxfordIIITPet(root='data', split='trainval', target_types='segmentation', 
                                     download=True,
                                     target_transform=lambda x: np.array(x.convert("L").resize((224, 224), Image.NEAREST)))
    test_ds_seg = OxfordIIITPet(root='data', split='test', target_types='segmentation',
                                 target_transform=lambda x: np.array(x.convert("L").resize((224, 224), Image.NEAREST)))
    
    # Merge datasets (same as get_data does)
    all_ds_seg = ConcatDataset([trainval_ds_seg, test_ds_seg])
    total_size = len(all_ds_seg)
    
    # Recreate the EXACT same shuffle and split as ws_utils.get_data()
    all_indices = list(range(total_size))
    random.seed(42)
    random.shuffle(all_indices)
    
    # Split with same ratios (70/15/15)
    train_ratio = 0.7
    val_ratio = 0.15
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_indices = all_indices[:train_size]
    
    # Create subset for train split
    train_dataset_seg = Subset(all_ds_seg, train_indices)
    
    print(f"Loaded {len(train_dataset_seg)} ground truth samples matching CAM generation\n")
    
    for i, img_file in enumerate(image_files):
        if i >= len(train_dataset_seg):
            print(f"⚠️  Warning: Not enough GT data for {img_file}, skipping")
            continue
            
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        
        # Load image and CAM
        img_path = os.path.join(images_dir, img_file)
        # Mask files are named mask_*.npy, not image_*.npy
        mask_filename = img_file.replace('image_', 'mask_').replace('.pt', '.npy')
        mask_path = os.path.join(masks_dir, mask_filename)
        
        image_tensor = torch.load(img_path)
        cam_heatmap = np.load(mask_path)
        cam_resized = cv2.resize(cam_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Get corresponding ground truth from the SAME dataset split
        # train_dataset_seg[i] returns (image, mask) tuple
        _, gt_mask = train_dataset_seg[i]
        gt_resized = gt_mask  # Already a numpy array, already 224x224
        
        # Generate visualization
        save_path = os.path.join(args.output_dir, f'pipeline_sample_{i+1}.png')
        visualize_wsss_pipeline(
            image_tensor,
            cam_resized,
            gt_resized,
            threshold=args.threshold,
            use_crf=args.use_crf,
            save_path=save_path
        )
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

