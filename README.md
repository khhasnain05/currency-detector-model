# 💸 Currency Note Detector AI 🇵🇰

An intelligent AI application that **detects and counts Pakistani currency notes** from uploaded images using a custom-built **Convolutional Neural Network (CNN)** in **PyTorch**. Just upload an image — and get the total amount detected with note breakdown in a neat table!

---

## ✨ Features

✅ Detects Pakistani currency notes:
- Rs. 10, 20, 50, 100, 500, 1000, 5000

✅ Upload image and get instant results  
✅ Beautiful and clean web interface (Gradio)  
✅ Displays note count in a structured table  
✅ Simple, fast, and accurate

---

## 📂 Folder Structure

```bash
currency-note-detector/
├── app.py # Gradio web interface
├── model.pth # Trained PyTorch model
├── images/ # Example input/output images
│ ├── input.jpg
│ └── result.png
├── README.md # Project documentation
├── requirements.txt # Python dependencies
```

---

## 🧠 Model Overview

- **Framework**: PyTorch
- **Model Type**: Convolutional Neural Network (CNN)
- **Classes**: 10, 20, 50, 100, 500, 1000, 5000 (PKR)
- **Training Data**: Labeled note images from multiple angles
- **Accuracy**: ~90% on validation dataset

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/khhasnain05/currency-detector-model.git
cd currency-note-detector
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Trained Model

Download the trained model.

### 4️⃣ Run the App
```bash
python app.py
```

Your browser will automatically open the Gradio interface.

---

## 📸 Example Output

![demo](https://github.com/user-attachments/assets/33e6e9e3-9f41-42b9-b447-8772fa7749d7)

---

## 🧰 Tech Stack

- 🐍  **Python**
- 🖼️  **OpenCV**
- 🧠  **PyTorch**
- 🌐  **Gradio Web Interface**

---

## 👨‍💻 Author
**Khawaja Hasnain**
