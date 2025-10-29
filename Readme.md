# 🌾 Cassava Crop Disease Detection

This project detects **cassava leaf diseases** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**, and provides an interactive **Streamlit web app** for easy image-based predictions.

---

## 📘 Overview

Cassava is a key food crop in many tropical regions, but its yield can be severely reduced by leaf diseases.  
This project leverages **deep learning** to automatically identify diseases from cassava leaf images.

The model is trained on the **Kaggle Cassava Leaf Disease Dataset** and can classify images into multiple disease types.

---

## 🧠 Features

- Detects cassava leaf diseases from uploaded images  
- Automatically trains or loads a pre-trained model (`crop_model.h5`)  
- Cleans CSV metadata to match available images  
- Simple and interactive **Streamlit** interface  
- Displays predicted disease name and confidence score  

---

## 📂 Project Structure

```
CROP-DISEASE-DETECTION/
│
├── images/                     # Folder containing cassava leaf images
├── app.py                      # Main Streamlit app file
├── class_names.json            # Class index to disease name mapping
├── crop_model.h5               # Trained CNN model
├── meta_deta.csv               # Original metadata
├── meta_deta_cleaned.csv       # Cleaned metadata
└── README.md                   # Project documentation
```

---

## 📊 Dataset

- **Dataset:** Cassava Leaf Disease Classification  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
- **Classes:**
  1. Cassava Bacterial Blight (CBB)  
  2. Cassava Brown Streak Disease (CBSD)  
  3. Cassava Green Mottle (CGM)  
  4. Cassava Mosaic Disease (CMD)  
  5. Healthy  

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/raghavPahwa27/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, manually install:
```bash
pip install streamlit tensorflow opencv-python pandas scikit-learn numpy
```

### 3️⃣ Add Dataset
Place your cassava leaf images inside the `images/` folder.  
Ensure `meta_deta.csv` correctly references those image files.

---

## 🚀 Running the App

Run the app with:
```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

---

## 🧩 Model Architecture

| Layer Type | Parameters |
|-------------|-------------|
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense | Softmax (output layer) |

---

## 📈 Training Details

- Input Image Size: **64×64**
- Optimizer: **Adam**
- Loss: **Categorical Crossentropy**
- Metric: **Accuracy**
- Epochs: **20**
- Train/Test Split: **80/20**

The trained model is saved as `crop_model.h5` for future predictions.

---

## 🖼️ Prediction Example

Upload an image of a cassava leaf, and the model predicts the disease along with the confidence score.

**Example Output:**
```
🧠 Predicted Disease: Cassava Mosaic Disease
Confidence: 96.48%
```

---

## 📚 References

- [Cassava Leaf Disease Classification - Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
- TensorFlow Documentation  
- Streamlit Documentation  

---

## 👨‍💻 Author

**Raghav Pahwa**  
🌐 [GitHub Profile](https://github.com/raghavPahwa27)
