"""
Unified Data Splitting Module
Provides consistent train/val/test split functionality for all modules
"""

import os
import random
import numpy as np
import torch
from typing import List, Tuple, Optional


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(
    data_list: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[List, List, List]:
    """
    Split data into train, validation, and test sets with specified ratios.
    
    Args:
        data_list (List): List of data items to split
        train_ratio (float): Ratio of training data (default: 0.7)
        val_ratio (float): Ratio of validation data (default: 0.15)
        test_ratio (float): Ratio of test data (default: 0.15)
        seed (Optional[int]): Random seed for reproducibility
        shuffle (bool): Whether to shuffle the data before splitting
        
    Returns:
        Tuple[List, List, List]: (train_data, val_data, test_data)
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or if data_list is empty
    """
    # Validate inputs
    if not data_list:
        raise ValueError("data_list cannot be empty")
    
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("All ratios must be non-negative, and train_ratio must be positive")
    
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create a copy and optionally shuffle
    data_copy = data_list.copy()
    if shuffle:
        random.shuffle(data_copy)
    
    # Calculate split indices
    total_size = len(data_copy)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split data
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]
    
    # Log split statistics
    print(f"\n{'='*60}")
    print(f"Data Split Summary (seed={seed if seed is not None else 'None'})")
    print(f"{'='*60}")
    print(f"Total samples: {total_size}")
    print(f"Train: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
    print(f"Val:   {len(val_data)} ({len(val_data)/total_size*100:.1f}%)")
    print(f"Test:  {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return train_data, val_data, test_data


def split_oxford_pet_dataset(
    trainval_file: str,
    test_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split Oxford-IIIT Pet dataset from trainval.txt and test.txt files.
    Creates a proper train/val/test split from the original trainval set.
    
    Args:
        trainval_file (str): Path to trainval.txt
        test_file (str): Path to test.txt
        train_ratio (float): Ratio for training data from trainval
        val_ratio (float): Ratio for validation data from trainval
        test_ratio (float): Ratio for test data from trainval (typically 0 if using original test set)
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[List[str], List[str], List[str]]: (train_names, val_names, test_names)
        
    Note:
        If test_ratio=0, the original test set from test.txt will be used.
        Otherwise, both trainval and test will be merged and re-split.
    """
    # Read files
    with open(trainval_file, 'r') as f:
        trainval_lines = f.read().strip().splitlines()
    with open(test_file, 'r') as f:
        test_lines = f.read().strip().splitlines()
    
    trainval_names = [line.split()[0] for line in trainval_lines]
    test_names_original = [line.split()[0] for line in test_lines]
    
    print(f"\nOriginal Oxford-IIIT Pet dataset:")
    print(f"  Trainval: {len(trainval_names)} images")
    print(f"  Test:     {len(test_names_original)} images")
    
    if test_ratio == 0:
        # Use original test set, split trainval into train and val
        print(f"\nUsing original test set, splitting trainval into train/val...")
        train_val_ratio = train_ratio / (train_ratio + val_ratio)
        val_val_ratio = val_ratio / (train_ratio + val_ratio)
        
        train_names, val_names, _ = split_data(
            trainval_names,
            train_ratio=train_val_ratio,
            val_ratio=val_val_ratio,
            test_ratio=0.0001,  # Small remainder
            seed=seed,
            shuffle=True
        )
        test_names = test_names_original
    else:
        # Merge everything and re-split
        print(f"\nMerging trainval and test, then re-splitting...")
        all_names = trainval_names + test_names_original
        train_names, val_names, test_names = split_data(
            all_names,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            shuffle=True
        )
    
    return train_names, val_names, test_names


def split_indices(
    total_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate train/val/test indices for a dataset of given size.
    
    Args:
        total_size (int): Total number of samples
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[List[int], List[int], List[int]]: (train_indices, val_indices, test_indices)
    """
    indices = list(range(total_size))
    return split_data(
        indices,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        shuffle=True
    )

