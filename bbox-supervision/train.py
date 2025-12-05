import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from crf import apply_crf

class ImageBBoxMaskDataset(Dataset):
    """
    PyTorch Dataset that loads images and bounding-box-derived masks. 
    Args:
        data_dir (str): Root directory containing 'images' and 'masks' subdirectories.
        image_subdir (str): Subfolder name for image tensors (.pt).
        mask_subdir (str): Subfolder name for mask files (.npy).
        image_jpg_dir (str): Path to original .jpg images for CRF refinement.
        percentage (float): Fraction of dataset to load (e.g. 1.0 = full dataset).
        use_crf (bool): Whether to apply CRF to refine masks (default True)
    """

    def __init__(self, data_dir, image_subdir='images', mask_subdir='masks',
                 image_jpg_dir=None, percentage=1.0, use_crf=True):
        self.image_dir = os.path.join(data_dir, image_subdir)  # .pt tensors
        self.mask_dir = os.path.join(data_dir, mask_subdir)    # .npy masks
        self.image_jpg_dir = image_jpg_dir or self.image_dir   # jpg source for CRF

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.pt')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files)

        self.use_crf = use_crf
        self.dataset_size = int(percentage * len(self.image_files))
        self.image_files = self.image_files[:self.dataset_size]
        self.mask_files = self.mask_files[:self.dataset_size]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image_tensor = torch.load(os.path.join(self.image_dir, self.image_files[idx])) 
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))              

        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        if self.use_crf:
            base_name = os.path.splitext(self.image_files[idx])[0]
            img_path = os.path.join(self.image_jpg_dir, base_name + ".jpg")
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.resize(img_cv2, (224, 224))

            fg = mask.astype(np.float32)
            bg = 1.0 - fg
            softmax_map = np.stack([bg, fg], axis=0)

            refined = apply_crf(img_cv2, softmax_map)
            mask = refined.astype(np.uint8)

        return image_tensor, torch.from_numpy(mask).float()

def run_training(config, val_percentage=0.15, patience=3, min_delta=0.001):
    """
    Train a DeepLabV3 model using weakly supervised masks with validation and early stopping.
    The model is trained using binary segmentation (foreground vs background) and saved to the path specified in the config.
    
    Args:
        config (dict): Configuration dictionary containing keys:
            - DEVICE: Device to use for training (e.g., "cuda" or "cpu").
            - MODEL_SAVE_PATH: Path to save the trained model.
            - TRAIN_DATA_DIR: Directory containing training data.
            - IMAGE_PATH: Path to original JPG images.
            - USE_CRF: Whether to use CRF-refined masks.
            - PERCENTAGE: Dataset usage ratio (e.g. 0.5 = use 50%).
            - BATCH_SIZE: Training batch size.
            - NUM_EPOCHS: Maximum number of training epochs.
            - LEARNING_RATE: Learning rate for optimizer.
        val_percentage (float): Percentage of training data to use for validation (default: 0.15).
        patience (int): Number of epochs to wait for improvement before early stopping (default: 3).
        min_delta (float): Minimum change in validation loss to be considered as improvement (default: 0.001).
    """

    device = torch.device(config["DEVICE"])

    model_name = config["MODEL_SAVE_PATH"]

    deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    for param in deeplab_model.backbone.parameters():
        param.requires_grad = False

    num_classes = 2 
    deeplab_model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes-1)
    deeplab_model = deeplab_model.to(device)

    optimizer = optim.Adam(deeplab_model.parameters(), lr=config["LEARNING_RATE"])
    criterion = nn.BCEWithLogitsLoss()

    # Load full dataset (mask_subdir and use_crf controlled by CONFIG)
    full_dataset = ImageBBoxMaskDataset(
        data_dir=config["TRAIN_DATA_DIR"],
        image_subdir="images",
        mask_subdir=config["MASK_SUBDIR"],  # Determined by USE_GRABCUT
        image_jpg_dir=config["IMAGE_PATH"],  # path to original jpgs
        percentage=config["PERCENTAGE"],
        use_crf=config["USE_CRF"]  # Determined by USE_CRF
    )

    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_percentage)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["SEED"])
    )

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    
    print(f"\nBounding Box segmentation training:")
    print(f"  Train samples:  {train_size}")
    print(f"  Val samples:    {val_size}")
    print(f"  Mask type:      {config['MASK_SUBDIR']}")
    print(f"  GrabCut:        {config.get('USE_GRABCUT', True)}")
    print(f"  CRF enabled:    {config['USE_CRF']}")
    print(f"  Max epochs:     {config['NUM_EPOCHS']}")
    print(f"  Early stop patience: {patience}\n")

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config["NUM_EPOCHS"]):
        # Training phase
        deeplab_model.train()
        running_loss = 0.0
        num_batches = 0

        for images, weak_labels in train_loader:
            images = images.to(device)
            weak_labels = weak_labels.to(device)
            optimizer.zero_grad()
            outputs = deeplab_model(images)['out']
            loss = criterion(outputs, weak_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        
        # Validation phase
        deeplab_model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, weak_labels in val_loader:
                images = images.to(device)
                weak_labels = weak_labels.to(device)
                outputs = deeplab_model(images)['out']
                loss = criterion(outputs, weak_labels.unsqueeze(1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deeplab_model.state_dict().copy()
            print(f"  → Validation loss improved to {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    # Save the best model
    if best_model_state is not None:
        deeplab_model.load_state_dict(best_model_state)
        print(f"\nRestoring best model (val_loss={best_val_loss:.4f})")

    torch.save(deeplab_model.state_dict(), model_name)
    print(f"Model saved as {model_name}\n")


