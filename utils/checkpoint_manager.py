"""
权重文件管理工具
负责保存、加载、命名模型权重
"""

import os
import torch
from datetime import datetime
import json
import shutil


class CheckpointManager:
    """
    管理模型权重的保存和加载
    
    文件命名格式:
    {model_name}_d{date}_e{epochs}_bs{batch_size}_lr{lr}_iou{iou:.3f}.pth
    
    Example:
    vgg16_d20250112_e5_bs32_lr0.0001_iou0.950.pth
    deeplabv3_crf_d20250112_e10_bs16_lr0.0001_iou0.680.pth
    """
    
    def __init__(self, save_dir="checkpoints"):
        """
        Args:
            save_dir: 保存权重的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model, config, metrics, model_name):
        """
        保存模型权重及其配置
        
        Args:
            model: PyTorch模型
            config: 训练配置字典 (epochs, batch_size, lr, etc.)
            metrics: 评估指标字典 (iou, accuracy, etc.)
            model_name: 模型名称 (vgg16, deeplabv3_crf, etc.)
        
        Returns:
            checkpoint_path: 保存的权重文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取关键超参数
        epochs = config.get('epochs', 'NA')
        batch_size = config.get('batch_size', 'NA')
        lr = config.get('learning_rate', 'NA')
        use_crf = config.get('use_crf', False)
        
        # 提取关键指标
        mean_iou = metrics.get('mean_iou', 0.0)
        
        # 构建文件名
        filename_parts = [
            model_name,
            f"d{timestamp}",
            f"e{epochs}",
            f"bs{batch_size}",
            f"lr{lr}",
            f"iou{mean_iou:.3f}"
        ]
        
        if 'deeplab' in model_name.lower() and use_crf:
            filename_parts.insert(1, "crf")
        
        filename = "_".join(filename_parts) + ".pth"
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        # 保存完整checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 保存配置为JSON（方便查看）
        config_path = checkpoint_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': config,
                'metrics': metrics,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Checkpoint saved successfully!")
        print(f"{'='*60}")
        print(f"File: {checkpoint_path}")
        print(f"Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        print(f"\nConfig:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
        print(f"{'='*60}\n")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model=None):
        """
        加载权重文件
        
        Args:
            checkpoint_path: 权重文件路径
            model: PyTorch模型（可选，如果不提供则只返回state_dict）
        
        Returns:
            如果提供了model: 返回加载好权重的model
            否则: 返回完整的checkpoint字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n{'='*60}")
        print(f"✓ Checkpoint loaded successfully!")
        print(f"{'='*60}")
        print(f"File: {checkpoint_path}")
        print(f"Model: {checkpoint.get('model_name', 'Unknown')}")
        print(f"Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        print(f"\nMetrics:")
        for key, value in checkpoint.get('metrics', {}).items():
            print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
        print(f"{'='*60}\n")
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            return checkpoint
    
    def list_checkpoints(self):
        """列出所有已保存的权重文件"""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        
        if not checkpoints:
            print("No checkpoints found.")
            return []
        
        print(f"\n{'='*60}")
        print(f"Available Checkpoints ({len(checkpoints)})")
        print(f"{'='*60}")
        
        for i, ckpt in enumerate(sorted(checkpoints), 1):
            size_mb = os.path.getsize(os.path.join(self.save_dir, ckpt)) / 1024 / 1024
            print(f"{i}. {ckpt} ({size_mb:.2f} MB)")
        
        print(f"{'='*60}\n")
        
        return checkpoints
    
    def get_best_checkpoint(self, model_name_filter=None, metric='mean_iou'):
        """
        获取最佳checkpoint
        
        Args:
            model_name_filter: 模型名称过滤（如'vgg', 'deeplab'）
            metric: 用于排序的指标（默认mean_iou）
        
        Returns:
            best_checkpoint_path: 最佳权重文件路径
        """
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        
        if model_name_filter:
            checkpoints = [f for f in checkpoints if model_name_filter in f.lower()]
        
        if not checkpoints:
            return None
        
        # 从文件名提取IoU（简单方法）
        best_ckpt = None
        best_score = -1
        
        for ckpt in checkpoints:
            try:
                # 从文件名解析IoU
                if 'iou' in ckpt:
                    iou_str = ckpt.split('iou')[-1].split('.pth')[0]
                    iou_value = float(iou_str)
                    if iou_value > best_score:
                        best_score = iou_value
                        best_ckpt = ckpt
            except:
                continue
        
        if best_ckpt:
            return os.path.join(self.save_dir, best_ckpt)
        else:
            return None


class ColabCheckpointManager(CheckpointManager):
    """
    Colab专用的Checkpoint Manager
    自动同步到Google Drive
    """
    
    def __init__(self, save_dir="checkpoints", drive_dir=None):
        """
        Args:
            save_dir: 本地保存目录
            drive_dir: Google Drive目录（如果在Colab）
        """
        super().__init__(save_dir)
        self.drive_dir = drive_dir
        
        # 如果提供了Drive目录，确保它存在
        if self.drive_dir:
            os.makedirs(self.drive_dir, exist_ok=True)
    
    def save_checkpoint(self, model, config, metrics, model_name):
        """保存权重并同步到Google Drive"""
        # 本地保存
        checkpoint_path = super().save_checkpoint(model, config, metrics, model_name)
        
        # 同步到Drive（如果配置了）
        if self.drive_dir:
            drive_path = os.path.join(self.drive_dir, os.path.basename(checkpoint_path))
            shutil.copy2(checkpoint_path, drive_path)
            
            # 同时复制config JSON
            config_path = checkpoint_path.replace('.pth', '_config.json')
            if os.path.exists(config_path):
                drive_config_path = os.path.join(self.drive_dir, os.path.basename(config_path))
                shutil.copy2(config_path, drive_config_path)
            
            print(f"✓ Checkpoint synced to Google Drive:")
            print(f"  {drive_path}\n")
        
        return checkpoint_path
    
    @staticmethod
    def mount_drive():
        """挂载Google Drive（仅在Colab中）"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted successfully!")
            return True
        except ImportError:
            print("⚠ Not running in Colab, Google Drive not mounted.")
            return False


def create_checkpoint_manager(use_colab=False, drive_dir=None):
    """
    工厂函数：创建合适的CheckpointManager
    
    Args:
        use_colab: 是否在Colab环境
        drive_dir: Google Drive目录
    
    Returns:
        CheckpointManager或ColabCheckpointManager实例
    """
    if use_colab:
        return ColabCheckpointManager(drive_dir=drive_dir)
    else:
        return CheckpointManager()


# ============= 使用示例 =============

if __name__ == "__main__":
    # 示例：保存VGG模型
    from torchvision import models
    
    # 创建checkpoint manager
    ckpt_manager = CheckpointManager(save_dir="checkpoints")
    
    # 模拟训练配置
    config = {
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'Adam',
        'use_crf': False
    }
    
    # 模拟评估指标
    metrics = {
        'mean_iou': 0.680,
        'fg_iou': 0.720,
        'bg_iou': 0.640,
        'train_acc': 0.950
    }
    
    # 创建模型
    model = models.vgg16(pretrained=True)
    
    # 保存checkpoint
    ckpt_path = ckpt_manager.save_checkpoint(
        model=model,
        config=config,
        metrics=metrics,
        model_name='vgg16'
    )
    
    # 列出所有checkpoints
    ckpt_manager.list_checkpoints()
    
    # 加载checkpoint
    loaded_model = ckpt_manager.load_checkpoint(ckpt_path, model=model)

