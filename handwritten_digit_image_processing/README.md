# Handwritten Digit Recognition using Machine Learning and Deep Learning

## Project Overview

This project focuses on recognizing handwritten digits (0–9) using the MNIST dataset. Multiple machine learning and deep learning models were implemented and compared to identify the best-performing classifier for digit recognition.

## Problem Statement

The objective of this project is to:

* Perform Exploratory Data Analysis (EDA) on the MNIST dataset.
* Classify handwritten digit images into one of the ten classes (0–9).
* Compare the performance of multiple classification models.
* Recommend the best model for production deployment.

## Dataset

The project uses the MNIST (Modified National Institute of Standards and Technology) dataset.

* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Number of Classes: 10 (Digits 0–9)

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Jupyter Notebook

## Data Preprocessing

* Normalized pixel values from 0–255 to 0–1.
* Reshaped image data for machine learning models.
* Prepared image tensors for CNN architecture.

## Models Implemented

1. K-Nearest Neighbors (KNN)
2. Random Forest Classifier
3. Artificial Neural Network (ANN)
4. Convolutional Neural Network (CNN)

## Model Performance

| Model         | Accuracy |
| ------------- | -------- |
| KNN           | 97%      |
| Random Forest | 97%      |
| ANN           | 97%      |
| CNN           | 98%      |

## Best Model

The Convolutional Neural Network (CNN) achieved the highest accuracy of 98% and was selected as the best model for production deployment. CNN performed better because it effectively extracts spatial features from image data.

## Challenges Faced

* Handling high-dimensional image data.
* Distinguishing visually similar handwritten digits.
* Managing training time for computationally intensive models.

These challenges were addressed through data normalization and the use of CNN architecture for improved feature extraction.

## Project Structure

```text
Handwritten-Digit-Recognition/
│
├── Handwritten.ipynb
├── README.md
└── requirements.txt
```

## Results

The project successfully classified handwritten digits and demonstrated that deep learning models, particularly CNNs, outperform traditional machine learning approaches for image recognition tasks.


