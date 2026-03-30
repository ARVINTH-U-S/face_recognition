# 🎯 Face Recognition & Tracking using KNN and YOLOv8

This project demonstrates a real-time face recognition system using the K-Nearest Neighbors (KNN) technique, combined with person tracking using YOLOv8. It enables identifying individuals from video streams and associating them with tracked body movements.

---

## 📂 Project Structure

```
.
├── face_recognition_with_knn_technique.py   # Face recognition using KNN
├── face_mapping_with_knn.py                 # Face + pose tracking
├── knn_examples/
│   └── train/
│       ├── Person_1/
│       │   ├── img1.jpg
│       │   ├── img2.jpg
│       ├── Person_2/
│       │   ├── img1.jpg
│       │   ├── img2.jpg
├── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2. Install dependencies

```bash
pip install opencv-python numpy face_recognition ultralytics
```
---

## 📸 Dataset Preparation

Organize your dataset in the following format:

```
knn_examples/train/
    ├── Person_A/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── Person_B/
    │   ├── img1.jpg
```

* Folder name = Person name (label)
* Add multiple images per person for better accuracy

---

## 🧠 How It Works

### 🔹 1. Face Recognition (KNN)

* Loads images from dataset
* Extracts facial encodings
* Uses distance-based KNN to classify faces
* Assigns label based on nearest neighbors

---

### 🔹 2. Face Mapping with Pose Tracking

* Detects faces and labels them
* Uses YOLOv8 pose model for tracking people
* Tracks individuals using IDs
* Maps face labels to tracked body using keypoints

---

## ▶️ Usage

### 🟢 Run Face Recognition

```bash
python face_recognition_with_knn_technique.py
```

**Output:**

* Displays labeled video

---

### 🔵 Run Face + Pose Tracking

```bash
python face_mapping_with_knn.py
```

**Output:**

* Displays pose tracking with labels

---
