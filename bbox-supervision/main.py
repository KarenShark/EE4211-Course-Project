import os
import cv2
import pickle
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

from split import generate_train_test_splits, set_seed as split_set_seed
from train import run_training
from evaluate import evaluate_segmentation
from bounding_box import set_seed as bb_set_seed, convert_selected_images_to_pt, save_all_masks_train
from testset import save_binary_masks_from_pickle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import download_data

# Determine base paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG = {
    "DEVICE": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    # Shared data paths (from project root)
    "IMAGE_PATH": os.path.join(PROJECT_ROOT, "data/images"),
    "MASK_PATH": os.path.join(PROJECT_ROOT, "data/annotations/xmls"),
    "TRIMAP_PATH": os.path.join(PROJECT_ROOT, "data/annotations/trimaps"),
    "GROUND_TRUTH_DIR": os.path.join(PROJECT_ROOT, "ground-truth"),
    # OEQ-specific paths (within bbox-supervision/)
    "TEST_IMGS_PATH": os.path.join(SCRIPT_DIR, "split/box_test_imgs.pickle"),
    "TRAIN_IMGS_PATH": os.path.join(SCRIPT_DIR, "split/box_train_imgs.pickle"),
    "VAL_IMGS_PATH": os.path.join(SCRIPT_DIR, "split/box_val_imgs.pickle"),
    "REFINED_MASKS_PATH": os.path.join(SCRIPT_DIR, "split/boundingbox_masks.pickle"),
    "MODEL_SAVE_PATH": os.path.join(SCRIPT_DIR, "box_trained.pth"),
    "MASK_SAVE_DIR": os.path.join(SCRIPT_DIR, "box-mask"),
    "TRAIN_DATA_DIR": os.path.join(SCRIPT_DIR, "box-mask/train"),
    "TEST_DATA_DIR": os.path.join(SCRIPT_DIR, "box-mask/test"),
    "GT_MASK_DIR": os.path.join(SCRIPT_DIR, "box-mask/test"),
    # Training parameters
    "SPLIT_RATE": 0.5,
    "N_CLASSES": 37, 
    "IMAGE_SIZE": (224, 224),
    "SEED": 42,  # Unified seed for fair comparison
    "MASK_THRESHOLD": 0.5,
    "PERCENTAGE": 0.3,
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "MASK_SUBDIR": "refined_masks",  # Will be overridden by command line args
    'EVAL_BATCH_SIZE': 8, 
    'USE_CRF': True,  # Will be overridden by command line args
    'USE_GRABCUT': True  # Will be overridden by command line args
}

def save_trimaps(dataset, save_dir):
    """
    Save trimap masks from a dataset as PNG images to the specified directory.
    Args:
        dataset (Dataset): A dataset object that returns image-mask pairs.
        save_dir (str): Directory path where the trimap images will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        trimap_img = Image.fromarray(mask_np.astype(np.uint8))
        trimap_file = os.path.join(save_dir, f"trimap_{i}.png")
        trimap_img.save(trimap_file)
        print(f"Saved trimap: {trimap_file}")

def main():
    """
    Main pipeline for training and evaluating a segmentation model using bounding box masks.
    Steps:
    1. Download dataset if not already present.
    2. Set random seeds for reproducibility.
    3. Generate or load train/test splits from XML annotations.
    4. Generate bounding box images and binary masks for training data.
    5. Generate binary masks from trimaps for test data.
    6. Train the segmentation model if not already trained.
    7. Evaluate the trained model on the test dataset.
    """

    #step0: check if data and ground truths are generated
    download_data()

    #step1: set seed
    split_set_seed(CONFIG["SEED"])
    bb_set_seed(CONFIG["SEED"])
    
    #step2: generate train:test split
    split_dir = os.path.dirname(CONFIG["TEST_IMGS_PATH"])
    os.makedirs(split_dir, exist_ok=True)
    if not os.path.exists(CONFIG["TEST_IMGS_PATH"]) or not os.path.exists(CONFIG["TRAIN_IMGS_PATH"]):
        print("Split files not found. Generating train/test split based on XML annotations...")
        generate_train_test_splits(CONFIG)
    else:
        print("Split files exist. Skipping generation of new splits.")

    #step3.1: generate bounding box images and binary masks if folder does not exist
    train_images_folder = os.path.join(SCRIPT_DIR, "box-mask/train/images")
    train_masks_folder = os.path.join(SCRIPT_DIR, f"box-mask/train/{CONFIG['MASK_SUBDIR']}")
    
    # Check if both images and required masks exist with matching counts
    images_exist = os.path.exists(train_images_folder) and len(os.listdir(train_images_folder)) > 0
    masks_exist = os.path.exists(train_masks_folder) and len(os.listdir(train_masks_folder)) > 0
    
    if not images_exist or not masks_exist:
        with open(CONFIG["TRAIN_IMGS_PATH"], 'rb') as f:
            train_imgs = pickle.load(f)
        print(f"Loaded {len(train_imgs)} train images from pickle file.")
        
        if not images_exist:
            print("Generating image tensors...")
            convert_selected_images_to_pt(train_imgs, CONFIG["IMAGE_PATH"], train_images_folder, image_size=CONFIG["IMAGE_SIZE"])
        
        # Always generate both types of masks to ensure availability
        print("Generating training masks (basic and refined)...")
        save_all_masks_train(train_imgs, CONFIG["MASK_PATH"], CONFIG["MASK_SAVE_DIR"], image_size=CONFIG["IMAGE_SIZE"])
    else:
        print(f"Train images and {CONFIG['MASK_SUBDIR']} already processed. Skipping processing.")

    #step3.2 generate ground truth images and binary masks  
    if not os.path.exists(CONFIG["TEST_DATA_DIR"]) or not os.listdir(CONFIG["TEST_DATA_DIR"]):
        with open(CONFIG["TEST_IMGS_PATH"], 'rb') as f:
            test_imgs = pickle.load(f)
        print(f"Loaded {len(test_imgs)} test images from pickle.")
        save_binary_masks_from_pickle(test_imgs, CONFIG["TRIMAP_PATH"], CONFIG["TEST_DATA_DIR"])
    else:
        print("Test folder already exists and is not empty, skipping binary mask generation.")

    #step4: train loss
    if not os.path.exists(CONFIG['MODEL_SAVE_PATH']):
        print(f"Training new model: {os.path.basename(CONFIG['MODEL_SAVE_PATH'])}")
        run_training(CONFIG)
    else:
        print(f"Model already exists: {os.path.basename(CONFIG['MODEL_SAVE_PATH'])}")
        print("Skipping training. Delete the model file to retrain.")

    #step5: evaluate on test set (binary seed masks)
    evaluate_segmentation(CONFIG)
  
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train bounding box based segmentation.')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help="Percentage of data to use (0-1). Use 0.1 for quick testing. Default: 1.0 (full dataset)")
    parser.add_argument('--use_grabcut', type=lambda x: x.lower() == 'true', default=True,
                        help="Use GrabCut to refine masks (True/False). Default: True")
    parser.add_argument('--use_crf', type=lambda x: x.lower() == 'true', default=True,
                        help="Use CRF to refine masks during training (True/False). Default: True")
    args = parser.parse_args()
    
    if not (0 < args.data_percentage <= 1):
        parser.error("--data_percentage must be between 0 and 1.")
    
    # Update CONFIG based on command line arguments
    CONFIG["DATA_PERCENTAGE"] = args.data_percentage
    CONFIG["PERCENTAGE"] = args.data_percentage  # Also update PERCENTAGE for training
    CONFIG["USE_GRABCUT"] = args.use_grabcut
    CONFIG["USE_CRF"] = args.use_crf
    
    # Set mask subdirectory based on GrabCut setting
    CONFIG["MASK_SUBDIR"] = "refined_masks" if args.use_grabcut else "basic_masks"
    
    # Generate unique model filename based on configuration
    model_suffix = ""
    if not args.use_grabcut:
        model_suffix = "_basic"  # No GrabCut, basic masks
    elif not args.use_crf:
        model_suffix = "_grabcut"  # GrabCut only
    else:
        model_suffix = "_grabcut_crf"  # Full pipeline
    
    CONFIG["MODEL_SAVE_PATH"] = os.path.join(SCRIPT_DIR, f"box_trained{model_suffix}.pth")
    
    # Also update results filename
    CONFIG["RESULTS_JSON"] = os.path.join(SCRIPT_DIR, f"results_bbox{model_suffix}.json")
    
    # Print configuration
    print(f"\n{'='*80}")
    print("Bounding Box Segmentation Configuration")
    print(f"{'='*80}")
    print(f"  Data percentage: {args.data_percentage*100:.0f}%")
    print(f"  Use GrabCut:     {args.use_grabcut}")
    print(f"  Use CRF:         {args.use_crf}")
    print(f"  Mask type:       {CONFIG['MASK_SUBDIR']}")
    print(f"  Model file:      {os.path.basename(CONFIG['MODEL_SAVE_PATH'])}")
    print(f"  Random seed:     {CONFIG['SEED']}")
    print(f"{'='*80}\n")
    
    main()