# 🧠 ASL Sign Language Detection App

This project is a real-time American Sign Language (ASL) recognition system that combines the power of **YOLOv8** (for hand detection), **MediaPipe** (for keypoint extraction), and a lightweight **MLP classifier** (for letter classification). The app runs on **Streamlit** with webcam support.

---

## 🚀 Features

- 🔍 **YOLOv8**: Detects the hand in real-time video frames.
- 🖥️ **Streamlit App**: Interactive UI for webcam-based live predictions..
- ✅ Modular and extensible for future model improvements.

---

## 📁 Dataset Used

**[ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**  
- 87,000 training images (200×200 px) in 29 labeled folders  
- Classes: A-Z, SPACE, DELETE, NOTHING

---

## 🧰 Requirements

Install dependencies in a virtual environment:

```bash
pip install -r requirements.txt

