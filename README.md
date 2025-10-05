# Ovarian Tumor Classification Using Transvaginal Ultrasound (TVS) Images

## 🩺 Project Overview
This project focuses on the classification of ovarian tumors using transvaginal ultrasound (TVS) images. It consists of a two-stage deep learning pipeline:
- 1. Segmentation – to localize the region of interest (ROI) from ultrasound images.
- 2. Classification – to categorize the type of ovarian tumor based on the segmented image.

## ✨ Key Features
- TVS ultrasound-based tumor analysis
- Deep learning–based segmentation for ROI extraction
- Multiclass classification of ovarian tumor types
- Built with Python 3.12 and PyTorch
- GPU acceleration supported for training and inference

## Requirements
- Python 3.12
- *requirement.txt


## Installation
```bash
pip install -r requirements.txt
```

## 🧠 Model Architecture
### Segmentation Model
The segmentation model identifies and extracts the ovarian region from TVS images, helping focus the classifier on relevant anatomical structures.

### Classification Model
The classification model receives the segmented region and predicts the tumor category, supporting multiple tumor types including:
- Chocolate cyst
- Serous cystadenoma
- Teratoma
- Theca cell tumor
- Mucinous cystadenoma
- Simple cyst
- High-grade serous carcinoma
- Normal ovary

## Usage
### Running the Segmentation Model
```python
# Load and preprocess ultrasound images
# Apply the trained segmentation model
```

### Running the Classification Model
```python
# Feed segmented outputs into the classification model
# Predict tumor categories
```

## Dataset
The experiments will be conducted with two datasets: 
- 1. MMOTU open source dataset
- 2. A local washu medical dataset

## License
This project is available under the MIT License.
