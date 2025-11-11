import os
import pickle
import random

def set_seed(seed):
    """
    Set the random seed for reproducibility. 
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def generate_train_test_splits(config, train_ratio=0.7, val_ratio=0.15):
    """
    Generate a train/val/test split based on available annotated XML files.
    Images with corresponding XML files are split into train/val sets.
    The rest of the images without annotations are used for testing.
    
    Args:
        config (dict): Configuration dictionary containing:
            - SEED (int): Random seed for shuffling.
            - MASK_PATH (str): Directory containing XML annotation files.
            - IMAGE_PATH (str): Directory containing image files.
            - TEST_IMGS_PATH (str): Output path to save list of test image paths (pickle).
            - TRAIN_IMGS_PATH (str): Output path to save list of train image paths (pickle).
            - VAL_IMGS_PATH (str): Output path to save list of validation image paths (pickle).
            - DATA_PERCENTAGE (float): Percentage of data to use (optional, default: 1.0).
        train_ratio (float): Ratio of training data from annotated images (default: 0.7).
        val_ratio (float): Ratio of validation data from annotated images (default: 0.15).
        
    Returns:
        Tuple: ((train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks))
    """

    # Set the seed for reproducibility.
    set_seed(config["SEED"])
    
    # Get data percentage (default to 1.0 if not specified)
    data_percentage = config.get("DATA_PERCENTAGE", 1.0)
    
    # List all XML files (annotations) in the MASK_PATH folder.
    xml_files = sorted([
        os.path.join(config["MASK_PATH"], name)
        for name in os.listdir(config["MASK_PATH"])
        if name.lower().endswith('.xml')
    ])
    
    annotated_pairs = []
    for xml_file in xml_files:
        base = os.path.splitext(os.path.basename(xml_file))[0]
        img_file = os.path.join(config["IMAGE_PATH"], base + '.jpg')
        if os.path.exists(img_file):
            annotated_pairs.append((img_file, xml_file))
        else:
            print(f"Warning: Image file for {xml_file} not found.")
    
    print(f"\nFound {len(annotated_pairs)} images with XML annotations")
    
    # Split annotated images into train and validation sets
    random.shuffle(annotated_pairs)
    total_annotated = len(annotated_pairs)
    
    # Calculate split sizes
    train_size = int(total_annotated * train_ratio / (train_ratio + val_ratio))
    val_size = total_annotated - train_size
    
    train_pairs = annotated_pairs[:train_size]
    val_pairs = annotated_pairs[train_size:]
    
    # Apply data_percentage if specified (for quick testing)
    if data_percentage < 1.0:
        keep_train = int(len(train_pairs) * data_percentage)
        keep_val = int(len(val_pairs) * data_percentage)
        train_pairs = train_pairs[:keep_train]
        val_pairs = val_pairs[:keep_val]
    
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []
    
    if train_pairs:
        train_imgs, train_masks = zip(*train_pairs)
        train_imgs = list(train_imgs)
        train_masks = list(train_masks)
    
    if val_pairs:
        val_imgs, val_masks = zip(*val_pairs)
        val_imgs = list(val_imgs)
        val_masks = list(val_masks)
    
    # All images without annotations go to test set
    full_img_paths = sorted([
        os.path.join(config["IMAGE_PATH"], name)
        for name in os.listdir(config["IMAGE_PATH"])
        if name.endswith('.jpg')
    ])
    
    test_imgs = [img for img in full_img_paths if img not in train_imgs + val_imgs]
    test_masks = []
    
    # Apply data_percentage to test set as well
    if data_percentage < 1.0:
        keep_test = int(len(test_imgs) * data_percentage)
        test_imgs = test_imgs[:keep_test]
    
    print(f"\nData split summary (seed={config['SEED']}):")
    print(f"  Train: {len(train_imgs)} images ({len(train_imgs)/total_annotated*100:.1f}% of annotated)")
    print(f"  Val:   {len(val_imgs)} images ({len(val_imgs)/total_annotated*100:.1f}% of annotated)")
    print(f"  Test:  {len(test_imgs)} images (no annotations)")
    
    # Save splits to pickle files
    with open(config["TEST_IMGS_PATH"], 'wb') as f:
        pickle.dump(test_imgs, f)
    with open(config["TRAIN_IMGS_PATH"], 'wb') as f:
        pickle.dump(train_imgs, f)
    
    # Save validation set
    val_imgs_path = config.get("VAL_IMGS_PATH", config["TRAIN_IMGS_PATH"].replace("train", "val"))
    with open(val_imgs_path, 'wb') as f:
        pickle.dump(val_imgs, f)
    
    print(f"\nSaved splits to:")
    print(f"  Train: {config['TRAIN_IMGS_PATH']}")
    print(f"  Val:   {val_imgs_path}")
    print(f"  Test:  {config['TEST_IMGS_PATH']}\n")
    
    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
