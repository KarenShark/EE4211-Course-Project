import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from utils import ImageCamMaskDataset

def train_deeplab(SAVE_MODEL_PATH, CAM_MASK_DIR, USE_CRF, num_epochs, device, 
                  val_percentage=0.15, patience=3, min_delta=0.001):
    """
    Train a DeepLabV3 model for binary segmentation using weak labels from CAM masks:
        - Loads a pretrained DeepLabV3 model with a modified classifier for binary output.
        - Trains it on weakly supervised data from CAM masks with validation monitoring.
        - Implements early stopping based on validation loss.
        - Saves the best model to disk.
    Args:
        SAVE_MODEL_PATH (str): File path to save the trained model weights.
        CAM_MASK_DIR (str): Directory containing CAM-based weak segmentation masks.
        USE_CRF (bool): Whether to use CRF-refined masks as training labels.
        num_epochs (int): Maximum number of training epochs.
        device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
        val_percentage (float): Percentage of data to use for validation (default: 0.15).
        patience (int): Number of epochs to wait for improvement before early stopping (default: 3).
        min_delta (float): Minimum change in validation loss to be considered as improvement (default: 0.001).
    """

    # Load the pretrained DeepLabV3 model
    deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    # Freeze all parameters in the backbone (ResNet-50) to reduce computational power needed
    for param in deeplab_model.backbone.parameters():
        param.requires_grad = False
    # Modify the final layer for binary segmentation
    num_classes = 2  # foreground or background
    deeplab_model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes-1)
    deeplab_model = deeplab_model.to(device)

    optimizer = optim.Adam(deeplab_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare dataset and split into train/val
    full_dataset = ImageCamMaskDataset(data_dir=CAM_MASK_DIR, percentage=1.0, use_crf=USE_CRF)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_percentage)
    train_size = dataset_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"\nDeepLab segmentation training:")
    print(f"  Train samples: {train_size}")
    print(f"  Val samples:   {val_size}")
    print(f"  CRF enabled:   {USE_CRF}")
    print(f"  Max epochs:    {num_epochs}")
    print(f"  Early stop patience: {patience}\n")

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        deeplab_model.train()
        running_loss = 0.0
        num_batches = 0
        
        for images, weak_labels in train_loader:
            images = images.to(device)
            weak_labels = weak_labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = deeplab_model(images)['out']  # Get the output of the model
            loss = criterion(outputs, weak_labels.unsqueeze(1))

            # Backward pass and optimize
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

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
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
    
    torch.save(deeplab_model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}\n")



