# Annotation Matters: Evaluating Grad-CAM and Bounding Box Weak Supervision under Modular Segmentation with CRF Refinement

## Project Overview
This project implements a weakly-supervised segmentation framework, comparing its performance to a fully-supervised baseline. Our approach uses weak supervision via bounding boxes and CAM-based methods, enhanced with CRF post-processing. The aim is to explore the trade-offs between weak and fully-supervised segmentation approaches and evaluate their efficacy with limited annotations.

## Part 1. Environment Setup (CPU-Compatible)
Below are the instructions to set up the project environment for CPU-only execution. This includes both the base conda environment and the additional packages required for weakly-supervised segmentation.

### Step 1: Create and Activate the Conda Environment
```
conda create -n comp0197-cw1-pt python=3.12 pip
conda activate comp0197-cw1-pt
```

### Step 2: Install PyTorch (CPU Version)
```
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Additional Packages

The extra packages needed for this project are:
- `opencv-python` for image I/O and preprocessing
- `matplotlib` for visualizing and saving predicted images
- `pydensecrf` for CRF post-processing (constrain-to-boundary loss)

```
pip install opencv-python
pip install matplotlib
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

All scripts in this repository have been tested to run smoothly in this CPU-compatible environment. Make sure you remain within this newly created comp0197-cw1-pt environment whenever you run the experiments.

## Part 2. Code Execution Instructions (Reproduce All Results)
Below is the full guide to reproduce all results presented in our submission. Each part corresponds directly to the structure of the COMP0197 coursework components (MRP & OEQ).

### Step 1: Weakly-Supervised Segmentation (MRP, main code)
This script initiates the weakly-supervised segmentation process using class activation maps (CAMs). The --use_crf True flag specifies that CRF post-processing will be applied to refine the segmentation boundaries. 

```
python weakly-supervised/main.py --use_crf True
```

### Step 2: Fully-Supervised Baseline (MRP, experiments)
This set of commands runs experiments for fully-supervised segmentation baselines using different architectures. In our report, we already concluded that the DeepLabV3+ ResNet-50 architecture had the best results among the tested models; therefore, it serves as the baseline model for comparison. The other two commands are hence not necessary but beneficial for the marker to understand why we came to such conclusions.
```
python fully-supervised/main.py --model_name fcn    #Optional 
python fully-supervised/main.py --model_name unet   #Optional 
python fully-supervised/main.py --model_name deeplab
```

### Step 3: Ablation Studies (MRP, experiments)
Two ablation studies have been conducted to investigate key components of our weakly-supervised segmentation framework:

1. **CRF Post-Processing Ablation:** In this study, the segmentation script is executed without CRF post-processing. The goal is to assess the impact of CRF on segmentation performance by comparing the refined segmentation outputs (with CRF) against those produced without any post-processing. 

```
python weakly-supervised/main.py --use_crf False
```

2. **Foreground Threshold Sensitivity**
We analyzed the effect of varying the binary threshold used during evaluation on the segmentation performance. Different thresholds were selected, but the default one is 0.05. To observe the effects of other thresholds, users can manually adjust the value when running the code, like 0.01, 0.1.
```
#baseline model 
python weakly-supervised/main.py --use_crf True --foreground_threshold 0.05

#the number here is optional and can be varied
python weakly-supervised/main.py --use_crf True --foreground_threshold 0.1 
```

### Step 4: Open-Ended Question: Bounding Box VS CAM-Based Weakly Supervised Segmentation (OEQ, experiments)
This script addresses an open-ended experimental question by comparing two annotation methods, where it evaluates segmentation performance using bounding box annotations versus CAM-based methods in Step 1.
```
python open-ended-question/main.py
```

