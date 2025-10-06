# 🧠 Criminal Detection using MTCNN & CNN

## 📌 Overview
This project implements an **automated facial recognition system** for **criminal detection** using **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection and a **Convolutional Neural Network (CNN)** for face classification.

The goal of this system is to detect faces from images or videos, extract them using MTCNN, and classify them as either *criminal* or *non-criminal* using a trained CNN model.

---

## 🚀 Features
- 🔍 Face detection using **MTCNN**
- 🧩 Face preprocessing (alignment, cropping, normalization)
- 🧠 Classification using **CNN**
- 📊 Evaluation with confusion matrix & classification report
- 🖼 Visualization of detection & prediction results
- 🎥 Optional real-time webcam detection

---

## 🧰 Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Face Detection | MTCNN |
| Data Handling | NumPy, OpenCV, Pillow |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Utility | tqdm |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
git clone https://github.com/MuhammadAuliaa/deteksiKriminal-MTCNN-CNN.git
cd criminal-detection-mtcnn-cnn

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Execute the file projects
MTCNN_CNN.ipynb & MTCNN_GAN_CNN.ipynb intented for training the data, while Testing.ipynb for classify and prediction the data