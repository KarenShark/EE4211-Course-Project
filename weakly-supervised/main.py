import torch 
import os 
import sys
import argparse
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import set_random_seed, get_data
from data import download_data
from config import Config
from vgg_train import train
from gen_cam_masks import generate_vgg_cam_masks
from train import train_deeplab
from evaluate import evaluate_deeplab_model
import shutil

def main(args):
    """
    Main function to execute the training and evaluation pipeline.
    Steps:
    1. Set random seed and device.
    2. Download and prepare dataset.
    3. Train a VGG classifier (if not already trained).
    4. Generate Grad-CAM masks using the trained classifier.
    5. Train a DeepLab segmentation model using CAM masks.
    6. Evaluate the trained segmentation model.

    Args:
        args (argparse.Namespace): Command-line arguments specifying CRF usage and threshold.
    """

    use_crf = args.use_crf == "True"
    foreground_threshold = args.foreground_threshold
    data_percentage = args.data_percentage
    
    if data_percentage < 1.0:
        print(f"\n{'='*80}")
        print(f"⚠️  QUICK TEST MODE: Using {data_percentage*100:.0f}% of data")
        print(f"{'='*80}\n")

    if use_crf:
        Config.DEEPLAB_MODEL_PATH = f"weakly-supervised/deeplabv3_crf.pth"
        Config.SAVE_PATH = f"weakly-supervised/pred_crf.png"
    else:
        Config.DEEPLAB_MODEL_PATH = f"weakly-supervised/deeplabv3_no_crf.pth"
        Config.SAVE_PATH = f"weakly-supervised/pred_no_crf.png"

    #step1: set seed
    set_random_seed(Config.SEED)

    #step2: set device
    device = Config.DEVICE
    print(f"Using device: {device}")

    # step3: download data
    download_data(data_path=Config.DATA_PATH)
    # copy downloaded data to default path used by torchvision.datasets.OxfordIIITPet to avoid re-downloading
    shutil.copytree(os.path.join(Config.DATA_PATH, "images"), os.path.join(Config.DATA_PATH, "oxford-iiit-pet/images"),
                    dirs_exist_ok=True)
    shutil.copytree(os.path.join(Config.DATA_PATH, "annotations"),
                    os.path.join(Config.DATA_PATH, "oxford-iiit-pet/annotations"),
                    dirs_exist_ok=True)

    #step4: train classifier if saved model does not exist
    if os.path.exists(Config.MODEL_PATH):
        print("Trained classifier model found. Skipping training.")
        model = torch.load(Config.MODEL_PATH)
    else:
        # Use new data split with validation
        train_loader, val_loader, test_loader = get_data(
            Config.VGG_TRAIN_BATCH_SIZE, 
            data_percentage=data_percentage,
            use_validation=True,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=Config.SEED
        )
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, Config.NUM_CLASSES)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        train(model, train_loader, val_loader, criterion, optimizer, device=Config.DEVICE, epochs=Config.VGG_TRAIN_EPOCHS)
        torch.save(model.state_dict(), Config.MODEL_PATH)

    #step5: generate cams 
    if os.path.exists(Config.OUTPUT_DIR):
        print("Processed images found. Skipping image generation.")
    else: 
        print(f"Generating Grad-CAM images.")
        generate_vgg_cam_masks(OUTPUT_DIR=Config.OUTPUT_DIR, MODEL_PATH=Config.MODEL_PATH, device='cpu', data_percentage=data_percentage)
        
    #step6: train loss if saved model does not exist
    if os.path.exists(Config.DEEPLAB_MODEL_PATH):
        print(f"Model file '{Config.DEEPLAB_MODEL_PATH}' found. Skipping training.")
    else:
        print(f"Model file '{Config.DEEPLAB_MODEL_PATH}' not found. Starting training...")
        train_deeplab(SAVE_MODEL_PATH=Config.DEEPLAB_MODEL_PATH,
                      CAM_MASK_DIR=Config.OUTPUT_DIR,
                      USE_CRF=use_crf,
                      num_epochs=Config.DEEPLAB_TRAIN_EPOCHS,
                      device=device)
        
    #step7: evaluate trained model
    evaluate_deeplab_model(MODEL_PATH=Config.DEEPLAB_MODEL_PATH,
                           device=device,
                           SAVE_PATH=Config.SAVE_PATH,
                           threshold=foreground_threshold)

if __name__ == '__main__':
    """
    Parses command-line arguments and starts the main training pipeline.
    Arguments:
        --use_crf (str): Whether to use CRF post-processing ('True' or 'False').
        --foreground_threshold (float): Threshold for foreground segmentation mask (default 0.05 if not specified)
    """

    parser = argparse.ArgumentParser(description="A training pipeline for VGG and DeepLab with Grad-CAM support.")
   
    parser.add_argument("--use_crf",
                        choices=['True', 'False'],
                        required=True,
                        help="Set to 'True' or 'False' to indicate whether to use CRF in training segmentation model.")

    parser.add_argument("--foreground_threshold",
                        type=float,
                        default=0.05,
                        help="Threshold value (between 0 and 1) for foreground segmentation.")
    
    parser.add_argument("--data_percentage",
                        type=float,
                        default=1.0,
                        help="Percentage of data to use (0-1). Use 0.1 for quick testing. Default: 1.0 (full dataset)")
    
    args = parser.parse_args()

    if not (0 <= args.foreground_threshold <= 1):
        parser.error("--foreground_threshold must be between 0 and 1.")
    
    if not (0 < args.data_percentage <= 1):
        parser.error("--data_percentage must be between 0 and 1.")

    main(args)