# Deep-Learning-Projects

A collection of deep learning and machine learning projects covering computer vision, time series forecasting, and image classification tasks, built as part of my ML/Data Science learning journey.

## Projects

### 🖐️ [IndiSignLang](./IndiSignLang) — Indian Sign Language Recognition
DataMites capstone project (PR-0019). A CNN trained from scratch to classify static Indian Sign Language hand signs (24 classes, A-Y excluding J) from images. Covers full EDA, sequential data splitting to avoid frame-leakage, preprocessing, and model evaluation.

**Tech:** Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
**Result:** ~74% test accuracy

---

### 🌾 [Riceleaf_Disease](./Riceleaf_Disease) — Rice Leaf Disease Classification
A CNN-based image classifier for detecting rice leaf diseases, aimed at early disease detection to reduce crop yield loss. Covers EDA, preprocessing, data augmentation, multiple CNN models, and performance comparison.

**Tech:** Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, OpenCV

---

### ✍️ [handwritten_digit_image_processing](./handwritten_digit_image_processing) — Handwritten Digit Recognition
Classifies handwritten digits (0-9) using the MNIST dataset, comparing multiple ML and deep learning models (KNN, Random Forest, ANN, CNN) to identify the best-performing classifier.

**Tech:** Python, TensorFlow/Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
**Result:** CNN achieved 98% accuracy, best among all models tested

---

### 🌡️ [Airtemp](./Airtemp) — Air Temperature Time Series Forecasting
DataMites internship project (PR-0019 / PRAICP-1003-AirTempTS). Forecasts monthly mean air temperature (Changi Climate Station, Singapore, 1982-2020) using SARIMA, chosen for its clear 12-month seasonal pattern. Includes stationarity testing, ACF/PACF analysis, and model validation.

**Tech:** Python, pandas, numpy, matplotlib, seaborn, statsmodels (SARIMAX), scikit-learn
**Result:** MAE 0.44, RMSE 0.50, MAPE 1.55%

## About

This repo documents hands-on practice with deep learning and machine learning fundamentals — image preprocessing, CNN architectures, time series forecasting, model training/evaluation, and proper ML workflow practices (train/val/test splitting, avoiding data leakage, EDA-driven decisions).

Large datasets are excluded from version control (see `.gitignore`) — each project's own README includes dataset details and download instructions where applicable.

## Author

**KK**
GitHub: [github.com/satyashry](https://github.com/satyashry)
LinkedIn: [linkedin.com/in/psatyashry24](https://linkedin.com/in/psatyashry24)
