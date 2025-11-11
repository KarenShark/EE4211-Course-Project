import os
import pickle
import time
import sys 
from random import shuffle
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import random 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings

#defined functions
import config
import models.deeplabv3plus
from dataset import SegmentationDataset, save_binary_masks
from models.fcn import FCN8s
from models.unet import UNet
from image import generate_segmentation_image
from result import evaluate_dataset_in_memory

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import download_data

def set_seed(seed):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(train_ratio=0.7, val_ratio=0.15, seed=None, data_percentage=1.0):
    """
    Load training, validation, and testing image and mask paths with proper train/val/test split.
    
    Now merges trainval and test data, then re-splits into 70/15/15 for standard CV practice.
    
    Args:
        train_ratio (float): Ratio of training data from all data (default: 0.7).
        val_ratio (float): Ratio of validation data from all data (default: 0.15).
        seed (int): Random seed for reproducibility (default: uses config.SEED).
        data_percentage (float): Percentage of data to use (0-1). Default: 1.0 (full dataset).
        
    Returns:
        tuple: (train_set, val_set, test_set) SegmentationDataset instances.
    """
    if seed is None:
        seed = config.SEED
        
    os.makedirs(os.path.dirname(config.TEST_IMGS_PATH), exist_ok=True)

    # Read trainval and test splits
    with open(config.TRAINVAL_FILE, 'r') as f:
        trainval_lines = f.read().strip().splitlines()
    with open(config.TEST_FILE, 'r') as f:
        test_lines = f.read().strip().splitlines()

    trainval_names = [line.split()[0] for line in trainval_lines]
    test_names_original = [line.split()[0] for line in test_lines]

    print(f"\nOriginal Oxford-IIIT Pet dataset:")
    print(f"  Trainval: {len(trainval_names)} samples")
    print(f"  Test:     {len(test_names_original)} samples")
    
    # Merge all data and re-split into 70/15/15
    all_names = trainval_names + test_names_original
    total_size = len(all_names)
    
    print(f"  Total:    {total_size} samples")
    print(f"\nMerging all data and re-splitting into {train_ratio*100:.0f}/{val_ratio*100:.0f}/{(1-train_ratio-val_ratio)*100:.0f}...")
    
    random.seed(seed)
    random.shuffle(all_names)
    
    # Calculate split sizes (70/15/15 of total data)
    test_ratio = 1.0 - train_ratio - val_ratio
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_names = all_names[:train_size]
    val_names = all_names[train_size:train_size + val_size]
    test_names = all_names[train_size + val_size:]
    
    # Apply data_percentage if specified (for quick testing)
    if data_percentage < 1.0:
        keep_train = int(len(train_names) * data_percentage)
        keep_val = int(len(val_names) * data_percentage)
        keep_test = int(len(test_names) * data_percentage)
        train_names = train_names[:keep_train]
        val_names = val_names[:keep_val]
        test_names = test_names[:keep_test]
    
    print(f"\nNew train/val/test split (seed={seed}):")
    print(f"  Train: {len(train_names)} ({len(train_names)/total_size*100:.1f}%)")
    print(f"  Val:   {len(val_names)} ({len(val_names)/total_size*100:.1f}%)")
    print(f"  Test:  {len(test_names)} ({len(test_names)/total_size*100:.1f}%)")

    # Create file paths
    train_imgs = [os.path.join(config.IMAGE_PATH, name + ".jpg") for name in train_names]
    train_masks = [os.path.join(config.MASK_PATH, name + ".png") for name in train_names]
    
    val_imgs = [os.path.join(config.IMAGE_PATH, name + ".jpg") for name in val_names]
    val_masks = [os.path.join(config.MASK_PATH, name + ".png") for name in val_names]
    
    test_imgs = [os.path.join(config.IMAGE_PATH, name + ".jpg") for name in test_names]
    test_masks = [os.path.join(config.MASK_PATH, name + ".png") for name in test_names]

    # Verify files exist
    for img_path in train_imgs + val_imgs + test_imgs:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
    
    for mask_path in train_masks + val_masks + test_masks:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")

    # Save test set info for later evaluation
    with open(config.TEST_IMGS_PATH, 'wb') as f:
        pickle.dump(test_imgs, f)
    with open(config.TEST_MASKS_PATH, 'wb') as f:
        pickle.dump(test_masks, f)

    # Create datasets with transforms
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor()
    ])

    train_set = SegmentationDataset(train_imgs, train_masks, transform)
    val_set = SegmentationDataset(val_imgs, val_masks, transform)
    test_set = SegmentationDataset(test_imgs, test_masks, transform)
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}\n")
    
    return train_set, val_set, test_set

def initialize_model(model_name: str, config):
    """
    Initialise a segmentation model based on the specified model name.

    Args:
        model_name (str): The name of the model to initialize (acceptable parameters: 'unet', 'fcn', 'deeplab').
        config (module): Configuration module with model parameters.

    Returns:
        torch.nn.Module: The initialized segmentation model.
    """
    allowed_backbones = ["unet", "fcn", "deeplab"]

    if not any(model_name.startswith(prefix) for prefix in allowed_backbones):
        raise ValueError(f"Model '{model_name}' not found. Please choose one of the allowed models: {', '.join(allowed_backbones)}")

    num_out = 1 if config.N_CLASSES == 2 else config.N_CLASSES

    if model_name.startswith("deeplab"):
        model = models.deeplabv3plus.modelling.__dict__['deeplabv3plus_resnet50'](
            num_classes=num_out,
            output_stride=config.OUTPUT_STRIDE
        ).to(config.DEVICE)
    elif model_name.startswith("fcn"):
        model = FCN8s(n_classes=num_out).to(config.DEVICE)
    elif model_name.startswith("unet"):
        model = UNet(config.ENC_CHANNELS, config.DEC_CHANNELS, num_out, config.IMAGE_SIZE).to(config.DEVICE)
    else:
        raise ValueError("Specified model not found. Please choose a valid model name (deeplab, fcn, unet).")

    print(f"Initialized model: {model_name} with {num_out} output channel(s)")
    return model

def train(model, train_loader, val_loader, patience=3, min_delta=0.001):
    """
    Train the segmentation model using training and validation datasets with early stopping.
    
    Args:
        model (torch.nn.Module): The segmentation model to train.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        patience (int): Number of epochs to wait for improvement before early stopping (default: 3).
        min_delta (float): Minimum change in validation loss to be considered as improvement (default: 0.001).
        
    Returns:
        dict: Training history with loss values for each epoch.
    """
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"\nStarting training with early stopping (patience={patience}, min_delta={min_delta})")
    print(f"Max epochs: {config.EPOCHS}\n")

    for epoch in range(config.EPOCHS):
        print(f'EPOCH: {epoch+1}/{config.EPOCHS}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, targets in train_loader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            pred = model(images)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
                pred = model(images)
                loss = loss_fn(pred, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'  Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  → Validation loss improved to {best_val_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'  → No improvement for {epochs_no_improve} epoch(s)')
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
    end_time = time.time() - start_time
    print(f'\nTotal training time: {end_time:.2f}s')
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model (val_loss={best_val_loss:.4f})\n')
    
    return history

    
def main(args):
    """
    Main workflow for training and evaluating a segmentation model.
    Steps:
    - Check for pre-trained model and evaluate if available.
    - Download data and generate binary masks.
    - Load data and initialize the model.
    - Train the model, save it, and run evaluation.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    data_percentage = args.data_percentage
    
    if data_percentage < 1.0:
        print(f"\n{'='*80}")
        print(f"⚠️  QUICK TEST MODE: Using {data_percentage*100:.0f}% of data")
        print(f"{'='*80}\n")
    
    model_name_mapping = {
        "deeplab": "deeplab",
        "unet": "unet",
        "fcn": "fcn"
    }

    if args.model_name in model_name_mapping:
        eval_model_name = model_name_mapping[args.model_name]
    else:
        print("Selected model not available. Only 'fcn', 'unet', or 'deeplab' are accepted.")
        return  


    if os.path.exists(config.MODEL_PATH):
        print("Trained model and history found. Skipping training and running evaluation.")

        generate_segmentation_image(config)

        evaluate_dataset_in_memory(
            model_name   = eval_model_name,
            model_path   = config.MODEL_PATH,
            test_imgs_pkl= config.TEST_IMGS_PATH,
            trimap_dir   = config.MASK_PATH,
            image_size   = config.IMAGE_SIZE,
            device       = config.DEVICE
        )
        return
    
    #step1: check whether data has been downloaded 
    download_data()

    #step2: save the binary masks from trimaps 
    save_binary_masks(config.MASK_PATH, config.BINARY_GROUND_TRUTH_DIR)

    #step3: prepare datasets with proper train/val/test split
    train_set, val_set, test_set = load_data(train_ratio=0.7, val_ratio=0.15, seed=config.SEED, data_percentage=data_percentage)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, drop_last=True, **config.KWARGS)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, **config.KWARGS)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, **config.KWARGS)

    #step4: train model with validation and early stopping
    model = initialize_model(args.model_name, config)
    history = train(model, train_loader, val_loader, patience=3, min_delta=0.001)
    torch.save(model, config.MODEL_PATH)
    print("Training complete. Model and history saved. Generating image and running evaluation now.")

    #step5: generate image
    generate_segmentation_image(config)

    #step6: compute metrics
    evaluate_dataset_in_memory(
        model_name   = eval_model_name,
        model_path   = config.MODEL_PATH,
        test_imgs_pkl= config.TEST_IMGS_PATH,
        trimap_dir   = config.MASK_PATH,
        image_size   = config.IMAGE_SIZE,
        device       = config.DEVICE
    )

if __name__ == '__main__':

    set_seed(config.SEED if hasattr(config, 'SEED') else 100)
    parser = argparse.ArgumentParser(description='Train segmentation network.')
    parser.add_argument('--model_name', type=str, default='deeplabv3plus_resnet50',
                        help="Specify the backbone to train (e.g., 'unet', 'fcn', 'deeplab').")
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help="Percentage of data to use (0-1). Use 0.1 for quick testing. Default: 1.0 (full dataset)")
    args = parser.parse_args()
    
    if not (0 < args.data_percentage <= 1):
        parser.error("--data_percentage must be between 0 and 1.")

    dynamic_model_path = os.path.join(config.OUTPUT_PATH, f"{args.model_name}_model.pth")
    dynamic_pred_plot_path = os.path.join(config.OUTPUT_PATH, f"{args.model_name}_pred.png")
    
    config.MODEL_PATH    = dynamic_model_path
    config.PRED_PLOT_PATH = dynamic_pred_plot_path

    main(args)