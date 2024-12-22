#  Deep Learning Final Project Group 11

## System Requirements

- **Operating System**: Linux Ubuntu 22.04.5
- **Python Version**: 3.10
- **Anaconda**: Installed
- **PyTorch Version**: 2.5.1
- **CUDA Version**: 12.4
- **Other Packages**: pandas, numpy, einops, albumentations, imbalanced-learn, optuna, scipy, tqdm, etc.

## How to Run

### 1. Create and Activate Conda Environment

First, create a new Conda environment and activate it:

```bash
conda create -n ml_project python=3.11
conda activate ml_project
```

### 2. Install Required Packages

Install the required Python packages in the activated Conda environment:

#### Install Basic Packages
```bash
conda install pandas numpy scikit-learn scipy tqdm -y
```

#### Install PyTorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### Install Other Required Packages
```bash
pip install imbalanced-learn optuna
```

### 3. Data Preparation

Place the dataset in the 'archive' folder under the project directory, including the following files:
The folder structure should be as follows:

```
project_directory/
├── model_13.ipynb
├── archive/
│   ├── train_512
│   ├── train_mask_512
│   └── val_512
│   └── val_mask_512
└── bcss_dataset.py
```

### 5.Dataset

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("whats2000/breast-cancer-semantic-segmentation-bcss")

print("Path to dataset files:", path)
```
[text](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss)

### 6. Run the Main Program

Execute model_13.ipynb in Jupyter Notebook

The program will automatically execute the following steps:
1. Data Loading and Preprocessing:
    * Load training and validation datasets using BCSSDataset class
    * Perform data augmentation using albumentations, including:
        * Random cropping (320x320)
        * Horizontal and vertical flips
        * 90-degree rotation
        * Brightness and contrast adjustment
        * Hue, saturation, and value adjustment
        * Gaussian noise, Gaussian blur, motion blur
        * Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

2. Model Architecture Design: The program implements 6 different deep learning model architectures:
    * UNet: Classic encoder-decoder architecture
    * nnUNet: Improved UNet with instance normalization and residual connections
    * MultiResUNet: UNet with multi-resolution feature extraction
    * TransUNet: Hybrid architecture combining Transformer and UNet
    * SwinUNet: UNet variant using Swin Transformer
    * DenseUNet: UNet architecture incorporating DenseNet features

3. Training and Optimization Process:
    * Uses AdamW optimizer
    * Implements CosineAnnealingLR learning rate scheduler
    * Implements early stopping mechanism with patience of 8 epochs
    * Uses cross-entropy loss function
    * Monitors multiple evaluation metrics during training:
        * IoU (Intersection over Union)
        * Dice coefficient
        * Accuracy

4. Model Evaluation and Comparison:
    * Records and compares for each model:
        * Training and validation loss
        * IoU score
        * Dice coefficient
        * Accuracy
        * Training time
        * Model parameter count

5. Visualization and Results Analysis:
    * Plots learning curve comparison graphs
    * Generates model performance comparison tables
    * Creates radar charts for multi-dimensional comparison
    * Exports comparison results to CSV file

### 7. View Results

After training completion, the training process and final model performance visualizations will be displayed in the last cell.

### Important Notes
* GPU Support: Ensure that an NVIDIA GPU is installed and CUDA is properly configured to accelerate model training. Use the nvidia-smi command to check GPU status.
* Resource Requirements: Training deep learning models may require substantial memory and computational resources. It's recommended to use a machine with sufficient GPU memory.
* Data Path: If data is located in a different path, modify the data path settings in the main program to ensure correct data loading.
* Dependency Versions: Ensure all package versions match this guide to avoid dependency conflicts. Pay special attention to PyTorch and CUDA version compatibility.