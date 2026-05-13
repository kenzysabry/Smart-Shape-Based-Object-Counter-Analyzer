# Smart Shape-Based Object Counter & Analyzer

## Overview

This project is a real-time Computer Vision system that detects, analyzes, and counts objects such as cars and bicycles using image processing and shape analysis techniques.

The system does not stop at object detection only. It also studies the internal structure of detected objects using:

* Edge Detection
* Contour Analysis
* Hough Transform
* Shape Understanding

The project was developed as part of a Computer Vision Final Project.

---

# Features

## Real-Time Object Detection

* Detect multiple objects simultaneously
* Draw bounding boxes around detected objects
* Display object labels on screen

## Image Processing Pipeline

For each detected object, the system applies:

* Grayscale Conversion
* Gaussian Blur
* Canny Edge Detection

## Shape Detection

Using Hough Transform, the system detects:

* Circles (e.g., bicycle wheels)
* Lines (e.g., car body structure)

## Object Understanding

The system uses detected shapes to better understand the object type:

* Bicycle → usually contains two circles
* Car → usually contains lines and circles

## Object Counting

* Counts cars and bicycles separately
* Prevents double counting
* Updates counters in real time

## Interactive Controls

Trackbars are included to dynamically adjust:

* Edge thresholds
* Hough Transform parameters

---

# Technologies Used

* Python
* OpenCV
* NumPy
* Computer Vision Techniques
* Image Processing

---

# Project Structure

```bash
Smart-Shape-Object-Counter/
│
├── main.py                # Main application
├── utils.py               # Helper functions
├── requirements.txt       # Required libraries
├── assets/                # Images or videos
├── outputs/               # Saved outputs
└── README.md              # Project documentation
```

---

# Processing Pipeline

## 1. Object Detection

The system first detects objects inside the video frame.

## 2. Preprocessing

Each detected object is processed using:

* Grayscale conversion
* Gaussian blur for noise reduction
* Canny edge detection

## 3. Shape Analysis

Hough Transform is used to detect:

* Circular shapes
* Straight lines

## 4. Object Classification

Objects are classified depending on their detected shapes.

## 5. Counting System

Objects are counted while avoiding duplicate counting.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/kenzysabry/Smart-Shape-Object-Counter.git
cd Smart-Shape-Object-Counter
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

```bash
python main.py
```

---

# Output Display

The application displays:

* Bounding boxes
* Object names
* Detected lines and circles
* Real-time counters
* Edge detection results

---

# Example Applications

* Smart traffic monitoring
* Vehicle counting systems
* Surveillance systems
* Intelligent transportation analysis

---

# Future Improvements

* Deep learning-based object detection
* More accurate classification models
* Speed estimation
* Multi-camera tracking
* Traffic analytics dashboard


# License

This project is for educational purposes only.
