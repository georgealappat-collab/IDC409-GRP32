# IDC409-GRP32
# Overview
This project focuses on the automatic identification of plant species using leaf images.
We developed and compared two machine learning pipelines — Convolutional Neural Network (CNN) and XGBoost Classifier — to classify leaves from five plant species based on their morphological characteristics.
The following species are included:
1.	Acer
2.	Quercus
3.	Salix alba
4.	Ulmus carpinifolia
5.	Ulmus glabra
# Objectives
1. Develop a robust leaf image classification pipeline.
2. Compare deep learning (CNN) with a classical ML approach (XGBoost).
3. Evaluate accuracy, precision, recall, and F1-score for both models.
4. Build an interpretable, reproducible model pipeline for plant species detection.
# Dataset
Source: https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/
Total images used= 375
Size of data:2.55 GB
# 1. Convolutional Neural Network (CNN)
Preprocessing Steps:
Background removal using rembg
Image resizing (224×224)
Data augmentation (rotation, brightness, contrast, flipping)
Methodology
Implemented using TensorFlow/Keras.
Architecture includes:
Convolutional, pooling, and dense layers
Batch normalization and dropout for regularization
Training-testing split: 70% training / 30% testing
Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy
# 2. XGBoost Classifier
Converted each image from BGR → HSV colour space using OpenCV.
Computed a 3D colour histogram with 8×8×8 bins (total 512 features per image).
Normalized and flattened the histogram into a single feature vector.
Captures overall colour composition of each leaf species.
Resized all images to 128×128 pixels for uniform input size.
Stored all feature vectors and corresponding labels as NumPy arrays for fast loading.
Training-testing split: 80% training / 20% testing
# Visualizations
Training Curves (accuracy/loss over epochs)
Confusion Matrix Heatmaps
Sample Predictions
# Dependencies
Install required packages using:
pip install -r requirements.txt
# Main requirements:
Tensorflow
xgboost
scikit-learn
numpy
pandas
matplotlib
seaborn
tqdm
rembg
# Contributors
Prathamesh Shelke –Data Preparation, Model Development and Execution of CNN
George - Data Preparation, Model Development and Execution of XGBoost
# License
This project is released under the MIT License — see the LICENSE file for details.
# Contact
For queries or collaboration: 
prathameshacademics@gmail.com 
ms23049@iisermohali.ac.in
