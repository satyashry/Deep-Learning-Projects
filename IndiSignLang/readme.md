# Indian Sign Language (ISL) Recognition

**DataMites™ Project Mentoring PR-0019 — Data Science Capstone Project**

An image classification project that uses a Convolutional Neural Network (CNN) to recognize static Indian Sign Language (ISL) hand signs from images.

---

## Business Case

Develop a Machine Learning model for **Indian Sign Language (ISL) recognition** to improve communication accessibility for the deaf and hard-of-hearing community in India.

## Project Goal

The project focuses on:

* Image preprocessing and data preparation
* Exploratory Data Analysis (EDA)
* Training a CNN-based image classification model
* Predicting Indian Sign Language hand signs
* Evaluating model performance using classification metrics

---

## Project Overview

This project builds an image classification pipeline for recognizing static Indian Sign Language hand signs.

The dataset contains **24 static sign classes**, representing letters from **A–Y**, excluding **J** and **Z**, because both signs require motion rather than a static hand pose.

The complete pipeline includes:

1. Dataset inventory and validation
2. Image preprocessing
3. Exploratory Data Analysis
4. Redundancy reduction through image subsampling
5. Sequential train/validation/test splitting
6. Image augmentation
7. CNN model training
8. Model evaluation
9. Confusion matrix and classification report analysis

The CNN is trained **from scratch using TensorFlow/Keras**, without a pretrained or transfer-learning backbone.

---

## Dataset

The dataset contains approximately **5,000 images** distributed across 24 classes.

### Classes

```text
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

> **J and Z are excluded** because both signs involve movement and cannot be reliably represented as a single static hand image.

### Download

The dataset can be downloaded from:

https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip

### Dataset Structure

After downloading and extracting the dataset:

```text
IndiSignLang/
├── Data/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── ...
│   └── Y/
├── ISL_Data_Preparation.ipynb
└── ISL_Final_Submission.ipynb
```

---

## Setup

### 1. Download the Dataset

Download and extract the dataset using the link above.

### 2. Place the Dataset

Place the extracted `Data/` directory inside the project directory:

```text
IndiSignLang/
├── Data/
├── ISL_Data_Preparation.ipynb
└── ISL_Final_Submission.ipynb
```

### 3. Generate the Dataset Splits

Run:

```text
ISL_Data_Preparation.ipynb
```

This notebook performs dataset inventory, image validation, subsampling, and sequential train/validation/test splitting.

It generates:

```text
Split_Data/
├── train/
├── val/
└── test/
```

### 4. Train and Evaluate the Model

After generating `Split_Data/`, run:

```text
ISL_Final_Submission.ipynb
```

This notebook performs EDA, preprocessing, CNN training, and model evaluation.

---

## Important Data Leakage Consideration

A major consideration in this project is **data leakage**.

Images within each class directory are sequential frames extracted from a single continuous recording session. Therefore, adjacent images can be extremely similar, often containing:

* The same hand
* The same signer
* Nearly identical camera angle
* Similar background
* Similar lighting
* Only minor changes between consecutive frames

A conventional random train/test split could therefore place near-identical frames into both the training and test sets.

This would result in an artificially high test accuracy that does not accurately represent real-world generalization.

### Mitigation Strategy

To reduce this risk, the project uses:

* **Image subsampling** to remove excessive frame redundancy
* **Sequential splitting by frame order** instead of random splitting
* A separate test segment from the recording sequence
* **Training-only augmentation** to introduce additional variation

The sequential split provides a more realistic evaluation than a random frame-level split for this particular dataset.

---

## Project Structure

```text
IndiSignLang/
│
├── Data/
│   ├── A/
│   ├── B/
│   ├── ...
│   └── Y/
│
├── Split_Data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── ISL_Data_Preparation.ipynb
├── ISL_Final_Submission.ipynb
├── isl_simple_cnn.h5
└── README.md
```

### File Description

| File / Directory             | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| `Data/`                      | Raw dataset organized by sign class                       |
| `Split_Data/`                | Generated train, validation, and test datasets            |
| `ISL_Data_Preparation.ipynb` | Dataset inventory, validation, subsampling, and splitting |
| `ISL_Final_Submission.ipynb` | EDA, preprocessing, CNN training, and evaluation          |
| `isl_simple_cnn.h5`          | Saved trained CNN model                                   |
| `README.md`                  | Project documentation                                     |

> `Data/` and `Split_Data/` are excluded from version control because of their size. They can be regenerated locally using the setup instructions above.

---

# Methodology

## 1. Data Preparation

The data preparation pipeline includes:

* Inventory of images in each class
* Image file validation
* Detection of corrupted files
* Verification of image properties
* Visual inspection of samples
* Image subsampling
* Sequential train/validation/test splitting

---

## 2. Exploratory Data Analysis

EDA is performed to understand the characteristics of the dataset.

The analysis includes:

* Class-wise image distribution
* Image dimensions
* File size distribution
* Sample image visualization
* Pixel intensity distribution
* RGB channel analysis

Example visualization:

```text
Class Distribution
        ↓
Image Properties
        ↓
Sample Image Grid
        ↓
Pixel Intensity Analysis
```

---

## 3. Image Preprocessing

Images are preprocessed before being passed to the CNN.

### Resize

All images are resized to:

```text
128 × 128 pixels
```

### Normalization

Pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
rescale = 1./255
```

### Data Augmentation

Training images are augmented using:

* Rotation
* Width shifting
* Height shifting
* Zoom
* Brightness variation

Augmentation is applied **only to the training dataset**.

### Horizontal Flip

Horizontal flipping is intentionally **not used**.

Mirroring a hand sign can change the semantic meaning of the sign or produce an incorrect representation.

---

# 4. CNN Model

A Convolutional Neural Network is trained from scratch using **TensorFlow/Keras**.

### Architecture

The model consists of three convolutional blocks:

```text
Input Image
    │
    ▼
Conv2D (16 filters)
    │
MaxPooling2D
    │
    ▼
Conv2D (32 filters)
    │
MaxPooling2D
    │
    ▼
Conv2D (64 filters)
    │
MaxPooling2D
    │
    ▼
Flatten
    │
Dense
    │
Dropout
    │
    ▼
Softmax Output
    │
    ▼
24 Sign Classes
```

### Training Configuration

| Component               | Configuration             |
| ----------------------- | ------------------------- |
| Framework               | TensorFlow / Keras        |
| Input Size              | 128 × 128                 |
| Convolutional Blocks    | 3                         |
| Filters                 | 16 → 32 → 64              |
| Output Classes          | 24                        |
| Optimizer               | Adam                      |
| Loss Function           | Categorical Cross-Entropy |
| Regularization          | Dropout                   |
| Early Stopping          | Enabled                   |
| Learning Rate Reduction | ReduceLROnPlateau         |
| Transfer Learning       | Not used                  |

---

# 5. Model Evaluation

The model is evaluated using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Test accuracy
* Confusion matrix
* Precision
* Recall
* F1-score
* Classification report

---

# Results

The trained CNN achieved approximately:

```text
Test Accuracy: ~74%
```

### Observations

The evaluation indicates that classes with fewer training examples, such as:

* H
* U
* V
* W

tend to have lower recognition performance.

This is consistent with the class distribution observed during EDA.

The confusion matrix also shows increased confusion between some visually similar hand shapes.

---

## Limitations

### 1. Limited Dataset Diversity

The images originate from a limited recording setup, resulting in relatively consistent:

* Signer
* Background
* Lighting
* Camera setup
* Hand positioning

Therefore, the model's ability to generalize to unseen environments is not fully validated.

### 2. Single-Signer / Recording Bias

The dataset does not provide sufficient diversity across different signers and recording conditions.

A model trained on more diverse signers would be more representative of real-world usage.

### 3. Static Sign Limitation

The project recognizes static hand poses only.

The letters **J and Z** are excluded because their signs involve motion.

### 4. Model Architecture

The CNN is relatively lightweight and is trained from scratch.

More advanced architectures, transfer learning, or larger datasets could potentially improve performance.

---

## Future Improvements

Potential improvements include:

* Collecting data from multiple signers
* Increasing dataset size
* Adding different backgrounds and lighting conditions
* Using multiple camera angles
* Testing on completely unseen signers
* Applying transfer learning with pretrained CNN architectures
* Hyperparameter optimization
* Using a larger and deeper CNN architecture
* Implementing real-time webcam-based ISL recognition
* Extending the system to recognize dynamic signs such as J and Z using video-based models

---

## Tech Stack

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* scikit-learn

### Data Processing

* NumPy
* Pandas
* Pillow (PIL)

### Visualization

* Matplotlib
* Seaborn

---

## Model

The trained model is saved as:

```text
isl_simple_cnn.h5
```

The model can be loaded using TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model

model = load_model("isl_simple_cnn.h5")
```

---

## Reproducibility

To reproduce the project:

```text
1. Download the dataset
        ↓
2. Extract the Data/ directory
        ↓
3. Place Data/ inside the project
        ↓
4. Run ISL_Data_Preparation.ipynb
        ↓
5. Generate Split_Data/
        ↓
6. Run ISL_Final_Submission.ipynb
        ↓
7. Train and evaluate the CNN
```

---

## Project Information

**Project:** Indian Sign Language Recognition
**Project ID:** PRAICP-1000-IndiSignLang
**Program:** DataMites Data Science Program
**Project Type:** Data Science Capstone Project
**Model:** CNN trained from scratch
**Number of Classes:** 24
**Test Accuracy:** ~74%

---


### GitHub

[satyashry](https://github.com/satyashry)
