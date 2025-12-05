import numpy as np
import torch
from torchvision import models
import json
import os

from ws_utils import get_data, visualize_prediction

def compute_iou_from_masks(pred_mask, gt_mask, class_id, ignore_label=3):
    """
    Calculate iou for a specific class, ignoring certain pixels.
    Args:
        pred_mask (Tensor): Predicted mask.
        gt_mask (Tensor): Ground truth mask.
        class_id (int): Class label to compute IoU for.
        ignore_label (int): Label value to ignore when computing IoU.
    Returns:
        float: IoU score for the specified class.
    """

    valid_mask = gt_mask != ignore_label
    pred = (pred_mask == class_id) & valid_mask
    gt = (gt_mask == class_id) & valid_mask

    intersection = (pred & gt).sum().item()
    union = ((pred | gt) & valid_mask).sum().item()

    return intersection / union if union != 0 else float('nan')


def evaluate_model_on_loader(model, dataloader, device, foreground_threshold):
    """
    Evaluate a segmentation model over a dataset using IoU metrics.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        dataloader (DataLoader): DataLoader for the test or validation set.
        device (str or torch.device): Device to run the model on.
        foreground_threshold (float): Threshold to binarize prediction probabilities.
    Returns:
        dict: Dictionary with lists of IoU scores for foreground, background, and their mean.
    """

    model.eval()
    iou_fgs, iou_bgs, mean_ious = [], [], []

    with torch.no_grad():
        for images, trimaps in dataloader:
            images = images.to(device)
            trimaps = trimaps.to(device)

            # Get predictions
            outputs = model(images)['out']
            outputs = torch.sigmoid(outputs)
            preds = (outputs >= foreground_threshold).long()

            for pred_mask, gt_mask in zip(preds, trimaps):
                pred_mask = pred_mask.squeeze()
                pred_mask = torch.where(pred_mask == 0, 2, pred_mask)
                iou_fg = compute_iou_from_masks(pred_mask, gt_mask, class_id=1)
                iou_bg = compute_iou_from_masks(pred_mask, gt_mask, class_id=2)
                mean_iou = np.nanmean([iou_fg, iou_bg])

                iou_fgs.append(iou_fg)
                iou_bgs.append(iou_bg)
                mean_ious.append(mean_iou)

    print("\n=== Overall ===")
    print(f"Mean FG IoU: {np.nanmean(iou_fgs):.4f}")
    print(f"Mean BG IoU: {np.nanmean(iou_bgs):.4f}")
    print(f"Mean IoU:    {np.nanmean(mean_ious):.4f}")

    return {
        'fg_iou': iou_fgs,
        'bg_iou': iou_bgs,
        'mean_iou': mean_ious
    }


def evaluate_deeplab_model(MODEL_PATH, device, SAVE_PATH, threshold, data_percentage=1.0):
    """
    Load a trained DeepLab model, evaluate it on a test dataset, and save a sample prediction.
    Args:
        MODEL_PATH (str): Path to the saved DeepLab model weights.
        device (str or torch.device): Device to load and evaluate the model.
        SAVE_PATH (str): Path to save a sample predicted mask visualization.
        threshold (float): Foreground probability threshold for mask binarization.
        data_percentage (float): Percentage of data to use for evaluation (default: 1.0).
    """

    FOREGROUND_THRESHOLD = threshold

    _, _, test_loader = get_data(batch_size=8, target_types='segmentation', data_percentage=data_percentage)

    deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    num_classes = 2 
    deeplab_model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes-1)
    deeplab_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    deeplab_model = deeplab_model.to(device)
    deeplab_model.eval()

    results = evaluate_model_on_loader(model=deeplab_model, dataloader=test_loader,
                                       device=device, foreground_threshold=FOREGROUND_THRESHOLD)

    # Save results to JSON
    # Check if CRF is enabled (avoid matching "no_crf")
    use_crf = (("_crf.pth" in MODEL_PATH.lower() or "deeplabv3_crf" in MODEL_PATH.lower()) 
               and "no_crf" not in MODEL_PATH.lower())
    result_filename = "results_crf.json" if use_crf else "results_no_crf.json"
    result_path = os.path.join("image-level-supervision", result_filename)
    
    result_data = {
        'method': 'Weakly-Supervised (with CRF)' if use_crf else 'Weakly-Supervised (without CRF)',
        'mean_fg_iou': float(np.nanmean(results['fg_iou'])),
        'mean_bg_iou': float(np.nanmean(results['bg_iou'])),
        'mean_iou': float(np.nanmean(results['mean_iou'])),
        'data_percentage': data_percentage,
        'threshold': FOREGROUND_THRESHOLD,
        'num_test_samples': len(test_loader.dataset)
    }
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n✓ Results saved to: {result_path}")

    #visualise
    image_tensor = test_loader.dataset[0][0]
    pred_mask = visualize_prediction(deeplab_model, image_tensor,
                                     foreground_threshold=FOREGROUND_THRESHOLD,
                                     target_mask=test_loader.dataset[0][1],
                                     save_img_path=SAVE_PATH)
    
    return result_data
