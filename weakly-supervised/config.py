import os
import torch

class Config:

    SEED = 42
    
    # Number of classes based on target types.
    NUM_CLASSES = 37
    
    # Directory for data: images, annotations, etc.
    DATA_PATH = './data'
    
    # Output results directory (for saving CAM images, masks, etc.)
    OUTPUT_DIR = 'weakly-supervised/vgg_cam_masks'
    MODEL_PATH = 'weakly-supervised/vgg.pth'
    DEEPLAB_MODEL_PATH = 'weakly-supervised/deeplabV3.pth'
    SAVE_PATH = 'weakly-supervised/pred.png'
    
    # Training parameters
    VGG_TRAIN_BATCH_SIZE = 32
    DEEPLAB_TRAIN_BATCH_SIZE = 16
    VGG_TRAIN_EPOCHS = 5  # Increased from 3 for better convergence
    LEARNING_RATE = 1e-4
    DEEPLAB_TRAIN_EPOCHS = 10  # Increased from 2, with early stopping for optimal performance
    
    # device to be used 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
