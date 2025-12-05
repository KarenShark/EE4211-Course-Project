# Scripts 文件夹说明

## 📊 visualize_pipeline.py

生成弱监督语义分割流程的中间结果可视化。

### 功能

生成包含以下步骤的可视化图：
1. **Original Image**: 原始输入图像
2. **Grad-CAM Heatmap**: VGG16 生成的 CAM 热力图
3. **CAM Overlay**: CAM 叠加在原图上
4. **Binary Mask**: 使用阈值（τ=0.05）二值化后的伪标签
5. **CRF Refined** (可选): CRF 后处理后的结果
6. **Ground Truth**: 真实标注（用于对比）

### 使用方法

#### 基础用法（默认生成5个样本）：
```bash
python scripts/visualize_pipeline.py
```

#### 自定义参数：
```bash
# 生成10个样本，使用 CRF
python scripts/visualize_pipeline.py --num_samples 10 --use_crf

# 指定 CAM 目录和输出目录
python scripts/visualize_pipeline.py \
  --cam_dir image-level-supervision/vgg_cam_masks \
  --output_dir output/pipeline_viz \
  --threshold 0.05 \
  --use_crf
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cam_dir` | str | `image-level-supervision/vgg_cam_masks` | CAM 掩码目录 |
| `--output_dir` | str | `output/pipeline_visualization` | 输出目录 |
| `--num_samples` | int | 5 | 生成样本数量 |
| `--threshold` | float | 0.05 | 二值化阈值 |
| `--use_crf` | flag | False | 是否应用 CRF |

### 输出示例

生成的图像文件：
```
output/pipeline_visualization/
├── pipeline_sample_1.png
├── pipeline_sample_2.png
├── pipeline_sample_3.png
├── pipeline_sample_4.png
└── pipeline_sample_5.png
```

每个图像展示完整的流程：
```
[Original] → [CAM] → [Overlay] → [Binary] → [CRF] → [GT]
```

### 在 Colab 中使用

```python
# 在 Colab 中运行
!python scripts/visualize_pipeline.py --num_samples 10 --use_crf

# 查看生成的图像
from IPython.display import Image, display
import os

output_dir = 'output/pipeline_visualization'
for i in range(1, 6):
    img_path = f'{output_dir}/pipeline_sample_{i}.png'
    if os.path.exists(img_path):
        print(f"Sample {i}:")
        display(Image(filename=img_path))
```

### 注意事项

1. **运行前提**: 必须先运行弱监督训练生成 CAM 掩码
2. **依赖**: 需要 `pydensecrf` (如果使用 `--use_crf`)
3. **内存**: 生成多个样本时注意内存使用

### 用于 Report

这些可视化非常适合用于报告中展示：
- **Figure 1**: Pipeline Overview (展示完整流程)
- **Figure 2**: CAM Quality Analysis (对比 CAM 和 GT)
- **Figure 3**: CRF Effect (对比 w/ 和 w/o CRF)

---

## 其他脚本

目前只有 `visualize_pipeline.py`，未来可能添加：
- `generate_tables.py`: 生成 LaTeX 表格
- `plot_training_curves.py`: 绘制训练曲线
- `analyze_failures.py`: 分析失败案例

