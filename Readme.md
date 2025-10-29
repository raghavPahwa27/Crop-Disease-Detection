# 🌾 Crop Disease Detection (Cassava)

This project detects **cassava crop diseases** using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.  
It comes with a **pre-trained model (`crop_model.h5`)** included in the repository, so you can directly run predictions without retraining.

---

## 🚀 Features
- 🧠 Deep learning-based Cassava leaf disease classifier
- 📊 Trained on the **Kaggle Cassava Leaf Disease Dataset**
- 🌐 Streamlit web interface for real-time predictions
- 💾 Pre-trained model included (`crop_model.h5`)
- 🧹 CSV cleaned automatically before training

---

## 🧩 Dataset
The dataset used is from Kaggle:  
🔗 [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

The dataset contains labeled images of cassava leaves categorized into multiple disease classes.

---

## 🧠 Model Architecture
The CNN model consists of:
- 2 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Fully connected Dense layers
- Dropout for regularization
- Softmax output layer for classification

Model file included: **`crop_model.h5` (19 MB)**

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/raghavPahwa27/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

Then open the local URL (usually `http://localhost:8501`) in your browser.

---

## 📸 Usage
- Upload a cassava leaf image (`.jpg`, `.jpeg`, `.png`)
- The app displays:
  - The uploaded image
  - The predicted disease name
  - Confidence percentage

---

## 📂 Repository Structure
```
├── app.py                 # Main Streamlit app
├── crop_model.h5          # Pre-trained CNN model (19 MB)
├── meta_deta.csv          # Original dataset metadata
├── meta_deta_cleaned.csv  # Cleaned dataset metadata
├── class_names.json       # Class name mapping
├── README.md              # Project documentation
└── .gitignore
```

---

## 🧑‍💻 Author
**Raghav Pahwa**  
📧 [GitHub Profile](https://github.com/raghavPahwa27)

---

## 📜 License
This project is open-source and available under the **MIT License**.
