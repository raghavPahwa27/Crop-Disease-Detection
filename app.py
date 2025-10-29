

import os
import cv2
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam




BASE_DIR = "D:/CROP-DISEASE-DETECTION"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "meta_deta.csv")
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "meta_deta_cleaned.csv")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.h5")





def clean_csv(original_csv, image_folder, cleaned_csv):
    df = pd.read_csv(original_csv)
    df_cleaned = df[df["image_id"].apply(lambda x: os.path.exists(os.path.join(image_folder, x.strip())))]
    df_cleaned.to_csv(cleaned_csv, index=False)
    print(f"✅ Cleaned CSV saved with {len(df_cleaned)} valid images.")
    return cleaned_csv




@st.cache_data
def load_data_from_csv(image_dir, metadata_path):
    df = pd.read_csv(metadata_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        image_name = row['image_id'].strip()
        img_path = os.path.join(image_dir, image_name)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(row['label'])

    return np.array(images), np.array(labels)




def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model





st.title("🌾 Crop Disease Detector")


clean_csv(CSV_PATH, IMAGE_DIR, CLEANED_CSV_PATH)


X, y = load_data_from_csv(IMAGE_DIR, CLEANED_CSV_PATH)

if len(X) == 0 or len(y) == 0:
    st.error("❌ No images loaded. Check your 'images/' folder and CSV.")
    st.stop()

X = X / 255.0  




with open(CLASS_NAMES_PATH, "r") as f:
    class_map = json.load(f)





y = np.array([int(label) for label in y])
y_encoded = to_categorical(y)




X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)





if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("✅ Pre-trained model loaded.")
else:
    st.warning("⚠ Training new model...")
    with st.spinner("Training model..."):
        model = build_model(X.shape[1:], y_encoded.shape[1])
        model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
        model.save(MODEL_PATH)
    st.success("✅ Model trained and saved.")






uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

  



    class_names = [v for k, v in sorted(class_map.items(), key=lambda item: int(item[0]))]

    st.image(img, caption="🖼 Uploaded Leaf", use_container_width=True)
    st.success(f"🧠 Predicted Disease: *{class_names[predicted_class]}*")
    st.info(f"Confidence: *{confidence:.2f}%*")


