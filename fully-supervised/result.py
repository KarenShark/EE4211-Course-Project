import os
import numpy as np
import torch
import pickle
import json
from PIL import Image
from torchvision import transforms
import torch.serialization
import matplotlib.pyplot as plt

from models import deeplabv3plus

def load_trained_model(model_name, model_path, device='cpu'):
    """
    Load a trained DeepLabV3+ segmentation model.
    Args:
        model_name (str): Model architecture (only 'deeplab' is supported).
        model_path (str): Path to the saved model file.
        device (str): Device to load the model onto ("cpu", "cuda", etc.).
    Returns:
        torch.nn.Module: The loaded and ready-to-evaluate model.
    """
    if not model_name.lower().startswith('deeplab'):
        raise ValueError(f"Only 'deeplab' model is supported. Got: {model_name}")

    with torch.serialization.safe_globals({"models.deeplabv3plus.DeeplabV3PlusResNet50": deeplabv3plus.deeplabv3_resnet50}):
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model.to(device)

def compute_iou(valid_mask, pred_binary, gt_binary, class_id):
    """
    Compute iou for a given class.
    Args:
        valid_mask (np.ndarray): Boolean array indicating valid pixels.
        pred_binary (np.ndarray): Predicted binary mask.
        gt_binary (np.ndarray): Ground truth binary mask.
        class_id (int): Class to compute IoU with 0 = background and 1 = foreground. 
    Returns:
        float: IoU score for the specified class.
    """
    pred_class = (pred_binary == class_id) & valid_mask
    gt_class   = (gt_binary == class_id) & valid_mask
    intersection = np.logical_and(pred_class, gt_class).sum()
    union       = np.logical_or(pred_class, gt_class).sum()
    if union == 0:
        return float('nan')
    return intersection / union

def compute_pixel_accuracy(valid_mask, pred_binary, gt_binary):
    """
    Compute pixel-wise accuracy for the image region.
    Args:
        valid_mask (np.ndarray): Boolean array indicating valid pixels.
        pred_binary (np.ndarray): Predicted binary mask.
        gt_binary (np.ndarray): Ground truth binary mask.
    Returns:
        float: Pixel accuracy.
    """
    correct = ((pred_binary == gt_binary) & valid_mask)
    total_valid= valid_mask.sum()
    if total_valid == 0:
        return float('nan')
    return correct.sum() / total_valid

def get_binary_and_valid_mask(trimap_array):
    """
    Convert a trimap array into a binary mask and valid pixel mask.
    Args:
        trimap_array (np.ndarray): Input trimap array (values: 1, 2, 3).
    Returns:
        tuple: (binary_mask, valid_mask)
            - binary_mask (np.ndarray): Foreground=1, Background=0, Unknown=1
            - valid_mask (np.ndarray): True where label is not unknown
    """
    valid_mask  = (trimap_array != 3)
    binary_mask = np.copy(trimap_array)
    binary_mask[trimap_array == 1] = 0
    binary_mask[trimap_array == 2] = 1
    binary_mask[trimap_array == 3] = 1
    return binary_mask, valid_mask

def predict_mask_in_memory(model, img_path, image_size=(224,224), device='cpu'):
    """
    Run inference on a single image and return the predicted binary mask.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        img_path (str): Path to the input image.
        image_size (tuple): Size to resize image before inference.
        device (str): Device for model inference.
    Returns:
        np.ndarray: Predicted binary mask with values {0, 1}.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)         
        probs  = torch.sigmoid(logits)       
        pred   = (probs >= 0.5).float()     

    return pred.squeeze().cpu().numpy()      

def evaluate_one_image_in_memory(model, img_path, trimap_path, image_size=(224,224), device='cpu'):
    """
    Evaluate model performance on a single image using IoU and pixel accuracy.
    Args:
        model (torch.nn.Module): Trained segmentation model.
        img_path (str): Path to the input image.
        trimap_path (str): Path to the corresponding ground truth trimap image.
        image_size (tuple): Size to resize images and masks.
        device (str): Device to run inference on.
    Returns:
        tuple: (bg_iou, fg_iou, mean_iou, pixel_accuracy)
    """

    pred_mask = predict_mask_in_memory(model, img_path, image_size, device)

    resize_tfm = transforms.Resize(image_size)
    trimap_img = Image.open(trimap_path).convert("L")
    trimap_img = resize_tfm(trimap_img)
    trimap_arr = np.array(trimap_img).astype(np.uint8)

    gt_binary, valid_mask = get_binary_and_valid_mask(trimap_arr)

    iou_bg = compute_iou(valid_mask, pred_mask, gt_binary, 0)
    iou_fg = compute_iou(valid_mask, pred_mask, gt_binary, 1)
    mean_iou = np.nanmean([iou_bg, iou_fg])
    pixel_acc = compute_pixel_accuracy(valid_mask, pred_mask, gt_binary)
    return iou_bg, iou_fg, mean_iou, pixel_acc

def evaluate_dataset_in_memory(model_name, model_path, test_imgs_pkl, trimap_dir, image_size=(224,224), device='cpu'):
    """
    Evaluate a trained segmentation model on a dataset stored in memory.
    Args:
        model_name (str): Model architecture (only 'deeplab' is supported).
        model_path (str): Path to the trained model.
        test_imgs_pkl (str): Pickle file containing test image paths.
        trimap_dir (str): Directory containing trimap ground truth masks.
        image_size (tuple): Size to resize images and masks.
        device (str): Device for model evaluation.
    Returns:
        dict: Dictionary with lists of evaluation metrics (IoUs and pixel accuracy).
    """
    
    #step1: load model 
    model = load_trained_model(model_name, model_path, device=device)

    #step2: get test images
    with open(test_imgs_pkl, 'rb') as f:
        test_images = pickle.load(f)

    #step3: evaluate test images
    iou_bgs, iou_fgs, mean_ious, pixel_accs = [], [], [], []
    for img_path in test_images:
        base_name   = os.path.splitext(os.path.basename(img_path))[0]
        trimap_path = os.path.join(trimap_dir, base_name + ".png")

        iou_bg, iou_fg, miou, pix_acc = evaluate_one_image_in_memory(
            model, img_path, trimap_path, image_size, device
        )
        iou_bgs.append(iou_bg)
        iou_fgs.append(iou_fg)
        mean_ious.append(miou)
        pixel_accs.append(pix_acc)

    mean_bg_iou = np.nanmean(iou_bgs)
    mean_fg_iou = np.nanmean(iou_fgs)
    overall_miou= np.nanmean(mean_ious)
    mean_pixacc = np.nanmean(pixel_accs)

    print("\n=== Final Summary ===")
    print(f" Mean BG IoU:   {mean_bg_iou:.4f}")
    print(f" Mean FG IoU:   {mean_fg_iou:.4f}")
    print(f" Overall mIoU:  {overall_miou:.4f}")
    print(f" Mean PixAcc:   {mean_pixacc:.4f}")
    
    # Save results to JSON
    result_data = {
        'method': f'Fully-Supervised ({model_name})',
        'mean_bg_iou': float(mean_bg_iou),
        'mean_fg_iou': float(mean_fg_iou),
        'mean_iou': float(overall_miou),
        'mean_pixel_accuracy': float(mean_pixacc),
        'num_test_samples': len(test_images)
    }
    
    result_path = os.path.join("fully-supervised/output", f"results_{model_name}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n✓ Results saved to: {result_path}")

    return {
        "bg_ious": iou_bgs,
        "fg_ious": iou_fgs,
        "miou_list": mean_ious,
        "pixel_acc_list": pixel_accs,
        "result_data": result_data
    }