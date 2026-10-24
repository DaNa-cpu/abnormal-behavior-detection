# Detection of abnormal behaviors in video sequences using deep learning techniques
### 1. Project Objective

The aim is to develop a deep learning system for detecting abnormal behaviors in surveillance videos using a Convolutional Autoencoder (CNN Autoencoder). Trained on normal video frames, the model will detect anomalies based on reconstruction error. The project evaluates whether this unsupervised approach can distinguish normal from abnormal events without requiring anomaly labels during training.

### 2. Dataset

The project will use the **UCSD Pedestrian Dataset – Ped2**, which contains surveillance videos recorded by a fixed camera. The dataset includes 16 training sequences with normal behavior and 12 test sequences containing both normal and abnormal events. Video frames will be extracted, resized, and normalized, optionally converted to grayscale. Test annotations will be used to evaluate performance.

**Dataset:** http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

### 3. Evaluation Metric

The main evaluation metric will be frame-level ROC-AUC. Each test frame will be assigned an anomaly score based on its reconstruction error, such as Mean Squared Error (MSE). These scores will be compared with ground-truth labels to calculate ROC-AUC.

### 4. Hardware Feasibility

Implementation will be carried out in Google Colab using Python and PyTorch or TensorFlow. GPU access will support efficient training of a moderate-sized CNN Autoencoder. Since UCSD Ped2 is relatively compact, the proposed model requires fewer computational resources than complex video-based architectures. False positives and missed anomalies will also be analyzed, with Precision, Recall, and F1-score reported if binary classification is required.
