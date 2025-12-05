"""
Google Drive Backup Utility
Automatically backup model checkpoints to Google Drive when training with full dataset.
"""

import os
import shutil
from datetime import datetime


def should_backup_to_gdrive(data_percentage=1.0, force=False):
    """
    Determine if model should be backed up to Google Drive.
    
    Args:
        data_percentage: Percentage of data used (0-1)
        force: Force backup regardless of data percentage
        
    Returns:
        bool: True if should backup
    """
    if force:
        return True
    
    # Only backup when using full dataset (100%)
    return data_percentage >= 0.99


def backup_to_gdrive(
    model_path,
    gdrive_base="/content/drive/MyDrive/EE4211-Models",
    experiment_name=None,
    metadata=None
):
    """
    Backup model checkpoint to Google Drive.
    
    Args:
        model_path: Path to the model file to backup
        gdrive_base: Base directory in Google Drive
        experiment_name: Name of the experiment (e.g., "weakly_crf", "fully_deeplab")
        metadata: Optional metadata dict to save alongside model
        
    Returns:
        str: Path to backed up file, or None if failed
    """
    try:
        # Check if running in Colab with Drive mounted
        if not os.path.exists("/content/drive"):
            print("⚠️  Google Drive not mounted. Skipping backup.")
            print("   To enable backup, run: drive.mount('/content/drive')")
            return None
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            backup_dir = os.path.join(gdrive_base, experiment_name, timestamp)
        else:
            backup_dir = os.path.join(gdrive_base, "checkpoints", timestamp)
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy model file
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        filename = os.path.basename(model_path)
        backup_path = os.path.join(backup_dir, filename)
        
        print(f"\n{'='*80}")
        print(f"📦 Backing up model to Google Drive")
        print(f"{'='*80}")
        print(f"Source: {model_path}")
        print(f"Destination: {backup_path}")
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Copy file
        shutil.copy2(model_path, backup_path)
        
        # Save metadata if provided
        if metadata:
            import json
            metadata_path = os.path.join(backup_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved: {metadata_path}")
        
        print(f"✅ Backup completed successfully!")
        print(f"{'='*80}\n")
        
        return backup_path
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None


def backup_results_to_gdrive(
    results_dir="output",
    gdrive_base="/content/drive/MyDrive/EE4211-Results",
    experiment_name=None
):
    """
    Backup entire results directory to Google Drive.
    
    Args:
        results_dir: Local results directory
        gdrive_base: Base directory in Google Drive
        experiment_name: Optional experiment name for subdirectory
        
    Returns:
        str: Path to backed up directory, or None if failed
    """
    try:
        if not os.path.exists("/content/drive"):
            print("⚠️  Google Drive not mounted. Skipping backup.")
            return None
        
        if not os.path.exists(results_dir):
            print(f"⚠️  Results directory not found: {results_dir}")
            return None
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            backup_dir = os.path.join(gdrive_base, experiment_name, timestamp)
        else:
            backup_dir = os.path.join(gdrive_base, timestamp)
        
        print(f"\n{'='*80}")
        print(f"📦 Backing up results to Google Drive")
        print(f"{'='*80}")
        print(f"Source: {results_dir}")
        print(f"Destination: {backup_dir}")
        
        # Copy directory
        shutil.copytree(results_dir, backup_dir, dirs_exist_ok=True)
        
        print(f"✅ Results backup completed!")
        print(f"{'='*80}\n")
        
        return backup_dir
        
    except Exception as e:
        print(f"❌ Results backup failed: {e}")
        return None


def mount_gdrive_if_not_mounted():
    """
    Attempt to mount Google Drive if running in Colab.
    
    Returns:
        bool: True if Drive is mounted or successfully mounted
    """
    try:
        # Check if already mounted
        if os.path.exists("/content/drive/MyDrive"):
            return True
        
        # Try to mount
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Verify mount
        if os.path.exists("/content/drive/MyDrive"):
            print("✅ Google Drive mounted successfully")
            return True
        else:
            print("⚠️  Google Drive mount verification failed")
            return False
            
    except ImportError:
        # Not running in Colab
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

