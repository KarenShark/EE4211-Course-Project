import os
import torch

if torch.cuda.is_available():
    DEVICE = "cuda" 
elif torch.backends.mps.is_available():
    DEVICE = "mps"  
else:
    DEVICE = "cpu" 

# DataLoader kwargs: optimize for GPU if available
# num_workers=2 for optimal performance with multiprocessing
if DEVICE == 'cuda':
    KWARGS = {'pin_memory': True, 'num_workers': 2}
elif DEVICE == 'mps':
    KWARGS = {'pin_memory': False, 'num_workers': 2}  # MPS doesn't support pin_memory
else:
    KWARGS = {}  # CPU: use default settings

# Seed (unified across all modules for fair comparison)
SEED = 42

# Data paths
IMAGE_PATH = 'data/images'
MASK_PATH = 'data/annotations/trimaps'

# Data attributes
IMAGE_SIZE = (256, 256)
N_CLASSES = 2
BATCH_SIZE = 8

# Model parameters
ENC_CHANNELS = (3, 64, 128, 256, 512)
DEC_CHANNELS = (512, 256, 128, 64)

# Train / test split rate
SPLIT_RATE = 0.5

# Learning parameters
LEARNING_RATE = 0.0001
EPOCHS = 10  # Increased from 2, with early stopping for optimal performance

# Output stride
OUTPUT_STRIDE = 8

# Output paths
OUTPUT_PATH = 'fully-supervised/output'
SPLIT_PATH = 'fully-supervised/split'
TRAIN_IMGS_PATH = os.path.join(SPLIT_PATH, 'train_imgs.pickle')
VAL_IMGS_PATH = os.path.join(SPLIT_PATH, 'val_imgs.pickle')
TEST_IMGS_PATH = os.path.join(SPLIT_PATH, 'test_imgs.pickle')
TEST_MASKS_PATH = os.path.join(SPLIT_PATH, 'test_masks.pickle')
MODEL_PATH = os.path.join(OUTPUT_PATH, 'deeplab_model.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'deeplab_history.pickle')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_PATH, 'history.png')
PRED_PLOT_PATH = os.path.join(OUTPUT_PATH, 'pred.png')
GROUND_TRUTH_DIR = 'ground-truth'
GROUND_TRUTH_FOLDER = 'ground-truth'
BINARY_GROUND_TRUTH_DIR = 'fully-supervised/binary-ground-truth'
BINARY_GROUND_TRUTH_FOLDER = 'fully-supervised/binary-ground-truth'
TRIMAP_PATH = 'data/annotations/trimaps'
TRAINVAL_FILE = 'data/annotations/trainval.txt'
TEST_FILE = 'data/annotations/test.txt'