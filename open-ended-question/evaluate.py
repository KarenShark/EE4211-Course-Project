import os
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

def compute_iou_from_masks(pred_mask, gt_mask, class_id, ignore_label=3):
    """
    Compute iou for a specific class while ignoring pixels labeled with ignore_label.
    Args:
        pred_mask (Tensor): Predicted segmentation mask.
        gt_mask (Tensor): Ground truth mask.
        class_id (int): Class ID to compute IoU for (e.g., 0 for background, 1 for foreground).
        ignore_label (int): Label to ignore during evaluation.
    Returns:
        float: IoU score for the specified class.
    """
    valid_mask = gt_mask != ignore_label
    pred = (pred_mask == class_id) & valid_mask
    gt = (gt_mask == class_id) & valid_mask
    intersection = (pred & gt).sum().item()
    union = ((pred | gt) & valid_mask).sum().item()
    if pred.sum() == 0 and gt.sum() == 0:
        return float('nan')
    return intersection / union if union != 0 else 0.0

def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Compute overall pixel accuracy between the predicted and ground truth masks.
    Args:
        pred_mask (Tensor): Predicted segmentation mask.
        gt_mask (Tensor): Ground truth mask.
    Returns:
        float: Pixel accuracy.
    """
    correct = (pred_mask == gt_mask).sum().item()
    total = pred_mask.numel()
    return correct / total

class EvaluationDataset(Dataset):
    """
    Custom Dataset for evaluation: Loads image paths from a pickle file and their corresponding ground truth
    trimap PNGs, applies transforms, and returns image-mask pairs.
    Args:
        pickle_path (str): Path to the pickle file with image paths.
        gt_mask_dir (str): Directory containing ground truth trimap PNGs.
        image_size (tuple): Desired (width, height) of the output images.
        image_transform (callable, optional): Transform to apply to input images.
        mask_transform (callable, optional): Transform to apply to masks.
    """
    def __init__(self, pickle_path, gt_mask_dir, image_size=(224, 224), image_transform=None, mask_transform=None):
        with open(pickle_path, 'rb') as f:
            all_image_paths = pickle.load(f)
        self.image_paths = [p for p in all_image_paths if os.path.exists(p)]
        self.gt_mask_dir = gt_mask_dir
        self.image_size = image_size
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieve an image and its binarized ground truth mask.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[Tensor, Tensor]: (image, binary mask)
        """
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.image_transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        gt_mask_path = os.path.join(self.gt_mask_dir, base_name + ".png")
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        else:
            gt_mask = Image.fromarray(gt_mask)
        gt_mask = self.mask_transform(gt_mask)
        gt_mask = (gt_mask * 255).squeeze().to(torch.int64)
        gt_mask = torch.where(gt_mask > 127, 
                              torch.tensor(1, dtype=torch.int64, device=gt_mask.device),
                              torch.tensor(0, dtype=torch.int64, device=gt_mask.device))
        return image, gt_mask

def get_eval_data_loader(pickle_path, gt_mask_dir, batch_size, image_size=(224, 224)):
    """
    Create a DataLoader for the evaluation dataset.
    Args:
        pickle_path (str): Path to the pickle file with image paths.
        gt_mask_dir (str): Directory containing ground truth trimap PNGs.
        batch_size (int): Batch size for loading data.
        image_size (tuple): Desired image resolution.
    Returns:
        DataLoader: DataLoader for evaluation.
    """
    dataset = EvaluationDataset(pickle_path, gt_mask_dir, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_model_on_loader(model, dataloader, device):
    """
    Evaluate a segmentation model using the given DataLoader. Computes IoU for foreground and background classes and pixel accuracy.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        dataloader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run inference on.
    Returns:
        dict: Dictionary containing 'fg_iou', 'bg_iou', 'mean_iou', and 'pixel_accuracy'.
    """
    model.eval()
    iou_foregrounds, iou_backgrounds, mean_ious, pixel_accs = [], [], [], []

    with torch.no_grad():
        for images, gt_masks in dataloader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            outputs = model(images)['out']
            outputs = torch.sigmoid(outputs)
            preds = (outputs >= 0.5).long().squeeze(1)

            for pred_mask, gt_mask in zip(preds, gt_masks):
                iou_fg = compute_iou_from_masks(pred_mask, gt_mask, class_id=1)
                iou_bg = compute_iou_from_masks(pred_mask, gt_mask, class_id=0)
                mean_iou = np.nanmean([iou_fg, iou_bg])
                pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)

                iou_foregrounds.append(iou_fg)
                iou_backgrounds.append(iou_bg)
                mean_ious.append(mean_iou)
                pixel_accs.append(pixel_acc)

    print("\n=== Evaluation Results ===")
    print(f"Mean FG IoU: {np.nanmean(iou_foregrounds):.4f}")
    print(f"Mean BG IoU: {np.nanmean(iou_backgrounds):.4f}")
    print(f"Overall Mean IoU: {np.nanmean(mean_ious):.4f}")
    print(f"Mean Pixel Accuracy: {np.nanmean(pixel_accs):.4f}")

    return {
        'fg_iou': iou_foregrounds,
        'bg_iou': iou_backgrounds,
        'mean_iou': mean_ious,
        'pixel_accuracy': pixel_accs
    }

def visualize_prediction_sample(model, dataloader, device, gt_trimaps_dir, idx=1, save_dir="open-ended-question/images"):
    """
    Visualize and save one prediction from the evaluation dataset using the trimap ground truth.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        dataloader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run inference on.
        gt_trimaps_dir (str): Directory containing the original trimap ground truth PNGs.
        idx (int): Index of the image to visualize from the batch.
        save_dir (str): Directory to save the visualization image.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        # Retrieve one batch from the DataLoader.
        for images, _ in dataloader:
            # Get the image tensor and its corresponding original image path from the dataset.
            dataset_obj = dataloader.dataset
            image_path = dataset_obj.image_paths[idx]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Construct the path to the corresponding ground truth trimap.
            gt_trimap_path = os.path.join(gt_trimaps_dir, base_name + ".png")
            if not os.path.exists(gt_trimap_path):
                print(f"Ground truth trimap not found for {base_name} in {gt_trimaps_dir}")
                return
            gt_trimap = Image.open(gt_trimap_path)

            # Get the input image and perform model prediction.
            image = images[idx].unsqueeze(0).to(device)
            output = model(image)['out']
            output = torch.sigmoid(output)
            pred_mask = (output >= 0.5).float().squeeze().cpu().numpy()

            # Process input image for visualization.
            image_np = images[idx].permute(1, 2, 0).cpu().numpy()
            image_np = np.clip((image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

            # Create an overlay: blend the predicted mask with the input image.
            overlay = image_np * 0.3
            mask = pred_mask > 0.5
            overlay[mask] = image_np[mask]

            # Create subplots.
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            axs[0].imshow(image_np)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(pred_mask, cmap='gray')
            axs[1].set_title("Predicted Mask")
            axs[1].axis("off")

            axs[2].imshow(gt_trimap)
            axs[2].set_title("Ground Truth")
            axs[2].axis("off")

            axs[3].imshow(overlay)
            axs[3].set_title("Predict Mask Overlay Image")
            axs[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), bbox_inches='tight')
            plt.show()
            break

def evaluate_segmentation(config):
    """
    Run the full evaluation pipeline using the provided configuration.
    Args:
        config (dict): Configuration dictionary with keys:
            - DEVICE
            - MODEL_SAVE_PATH
            - TEST_IMGS_PATH
            - GT_MASK_DIR
            - GROUND_TRUTH_DIR
            - EVAL_BATCH_SIZE
            - IMAGE_SIZE
    Returns:
        dict: Evaluation results including IoUs and pixel accuracy.
    """
    device = torch.device(config["DEVICE"])
    model_path = config["MODEL_SAVE_PATH"]

    test_loader = get_eval_data_loader(
        pickle_path=config["TEST_IMGS_PATH"],
        gt_mask_dir=config["GT_MASK_DIR"],
        batch_size=config["EVAL_BATCH_SIZE"],
        image_size=config["IMAGE_SIZE"]
    )

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    results = evaluate_model_on_loader(
        model=model,
        dataloader=test_loader,
        device=device
    )

    visualize_prediction_sample(model, test_loader, device, gt_trimaps_dir=config["GROUND_TRUTH_DIR"])
    return results
