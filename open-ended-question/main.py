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

CONFIG = {
    "DEVICE": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    "IMAGE_PATH": "data/images",               
    "MASK_PATH": "data/annotations/xmls",
    "GROUND_TRUTH_DIR": 'ground-truth',
    "TEST_IMGS_PATH": "open-ended-question/split/box_test_imgs.pickle",
    "TRAIN_IMGS_PATH": "open-ended-question/split/box_train_imgs.pickle",
    "VAL_IMGS_PATH": "open-ended-question/split/box_val_imgs.pickle",
    "REFINED_MASKS_PATH": "open-ended-question/split/boundingbox_masks.pickle",
    "MODEL_SAVE_PATH": "open-ended-question/box_trained.pth",
    "MASK_SAVE_DIR": "open-ended-question/box-mask",
    "TRAIN_DATA_DIR": "open-ended-question/box-mask/train",
    "TEST_DATA_DIR": "open-ended-question/box-mask/test",
    "SPLIT_RATE": 0.5,
    "N_CLASSES": 37, 
    "IMAGE_SIZE": (224, 224),
    "SEED": 50,
    "MASK_THRESHOLD": 0.5,
    "PERCENTAGE": 0.3,
    "NUM_EPOCHS": 10,  # Increased from 2, with early stopping for optimal performance
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "MASK_SUBDIR": "refined_masks", 
    'TRIMAP_PATH': 'data/annotations/trimaps',
    'GT_MASK_DIR': 'open-ended-question/box-mask/test',
    'EVAL_BATCH_SIZE': 8, 
    'USE_CRF': True
}

CONFIG["GROUND_TRUTH_DIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ground-truth")

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
    train_images_folder = "open-ended-question/box-mask/train/images"
    if not os.path.exists(train_images_folder) or not os.listdir(train_images_folder):
        with open(CONFIG["TRAIN_IMGS_PATH"], 'rb') as f:
            train_imgs = pickle.load(f)
        print(f"Loaded {len(train_imgs)} train images from pickle file.")
        convert_selected_images_to_pt(train_imgs, CONFIG["IMAGE_PATH"], "open-ended-question/box-mask/train/images", image_size=CONFIG["IMAGE_SIZE"])
        save_all_masks_train(train_imgs, CONFIG["MASK_PATH"], CONFIG["MASK_SAVE_DIR"], image_size=CONFIG["IMAGE_SIZE"])
    else:
        print("Train images and masks already processed. Skipping processing.")

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
        run_training(CONFIG)
    else:
        print("DeepLab model already trained and saved.")

    #step5: evaluate on test set (binary seed masks)
    evaluate_segmentation(CONFIG)
  
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train bounding box weakly-supervised segmentation.')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help="Percentage of data to use (0-1). Use 0.1 for quick testing. Default: 1.0 (full dataset)")
    args = parser.parse_args()
    
    if not (0 < args.data_percentage <= 1):
        parser.error("--data_percentage must be between 0 and 1.")
    
    if args.data_percentage < 1.0:
        print(f"\n{'='*80}")
        print(f"⚠️  QUICK TEST MODE: Using {args.data_percentage*100:.0f}% of data")
        print(f"{'='*80}\n")
        CONFIG["DATA_PERCENTAGE"] = args.data_percentage
    else:
        CONFIG["DATA_PERCENTAGE"] = 1.0
    
    main()