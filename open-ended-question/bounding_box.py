import os
import cv2
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

def set_seed(seed):
    """
    Set random seeds for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_annotation(xml_file):
    """
    Process an XML annotation file and extract bounding boxes.
    Args:
        xml_file (str): Path to the XML annotation file.
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def create_basic_mask(image_shape, boxes):
    """
    Create a binary mask with 1 as foreground and 0 as background
    Args:
        image_shape (tuple): Shape of the original image.
        boxes (list): List of bounding boxes.
    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for (xmin, ymin, xmax, ymax) in boxes:
        mask[ymin:ymax, xmin:xmax] = 1  
    return mask

def save_basic_masks(image_paths, annotation_dir, save_dir, image_size=(224, 224)):
    """
    Generate and save binary masks based on bounding boxes from annotations.
    Args:
        image_paths (list): List of image file paths.
        annotation_dir (str): Directory containing XML annotation files.
        save_dir (str): Output directory to save .npy masks.
        image_size (tuple): Desired size of output masks.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for img_path in image_paths:
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            xml_file = os.path.join(annotation_dir, f"{base_name}.xml")
            
            if not os.path.exists(xml_file):
                print(f"Skipping {img_path}: annotation not found.")
                continue
            
            boxes = parse_annotation(xml_file)
            basic_mask = create_basic_mask(image.shape, boxes)
            mask_resized = cv2.resize(basic_mask, image_size, interpolation=cv2.INTER_NEAREST)
            
            np.save(os.path.join(save_dir, f"{base_name}.npy"), mask_resized)
        
        except Exception as e:
            pass

def refine_mask_with_grabcut(image, rect):
    """
    Apply GrabCut to refine the foreground mask with bounding boxes. 
    Args:
        image (np.ndarray): Input image.
        rect (tuple): Bounding box (x, y, width, height) for initialization.
    Returns:
        np.ndarray: Refined binary mask.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return mask_refined

def save_refined_masks(image_paths, annotation_dir, save_dir, image_size=(224, 224)):
    """
    Generate and save GrabCut-refined masks with bounding box annotations
    Args:
        image_paths (list): List of image file paths.
        annotation_dir (str): Directory containing XML annotation files.
        save_dir (str): Output directory to save .npy refined masks.
        image_size (tuple): Desired size of output masks.
    """
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
    
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            xml_file = os.path.join(annotation_dir, f"{base_name}.xml")
            
            if not os.path.exists(xml_file):
                print(f"Skipping {img_path}: annotation not found.")
                continue
            
            boxes = parse_annotation(xml_file)
            if not boxes:
                raise ValueError(f"No boxes found in {xml_file}")
            
            # Use the first bounding box for GrabCut refinement.
            xmin, ymin, xmax, ymax = boxes[0]
            rect = (xmin, ymin, xmax - xmin, ymax - ymin)
            refined_mask = refine_mask_with_grabcut(image, rect)
            mask_resized = cv2.resize(refined_mask, image_size, interpolation=cv2.INTER_LINEAR)
            
            np.save(os.path.join(save_dir, f"{base_name}.npy"), mask_resized)
        
        except Exception as e:
            pass

def convert_selected_images_to_pt(image_paths, jpg_base_dir, pt_out_dir, image_size=(224, 224)):
    """
    Convert images to normalised PyTorch tensors and save as .pt files.
    Args:
        image_paths (list): List of full paths to selected images.
        jpg_base_dir (str): Base directory where original JPEG images are located.
        pt_out_dir (str): Directory to save the .pt tensor files.
        image_size (tuple): Size to which images should be resized.
    """
    os.makedirs(pt_out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for full_path in image_paths:
        base_name = os.path.basename(full_path)
        base_no_ext = os.path.splitext(base_name)[0]
        jpg_path = os.path.join(jpg_base_dir, base_name)
        if not os.path.exists(jpg_path):
            print(f"Image not found: {jpg_path}")
            continue

        img = Image.open(jpg_path).convert("RGB")
        tensor = transform(img)
        torch.save(tensor, os.path.join(pt_out_dir, base_no_ext + ".pt"))


def save_all_masks_train(train_imgs, annotation_dir, out_root, image_size=(224, 224)):
    """
    Called function in main that generates and saves both basic and GrabCut-refined masks for training images.
    Args:
        train_imgs (list): List of training image paths.
        annotation_dir (str): Directory containing XML annotation files.
        out_root (str): Root output directory to save both mask types.
        image_size (tuple): Desired size for saved masks.
    """

    print("Saving basic masks for training set...")
    train_basic_save_dir = os.path.join(out_root, "train/basic_masks")
    save_basic_masks(train_imgs, annotation_dir, train_basic_save_dir, image_size)

    print("Saving refined masks for training set...")
    train_refined_save_dir = os.path.join(out_root, "train/refined_masks")
    save_refined_masks(train_imgs, annotation_dir, train_refined_save_dir, image_size)

    print("All training masks saved.")
