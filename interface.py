import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import time
import os
from PIL import Image
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import cv2
import random
from utils import show_all_data_mtcnn, show_preprocessed_samples, predict_image_streamlit, preprocess_mtcnn_results

# Direktori dataset
data_dir = "data"
categories = ["criminal-v1", "non-criminal-v1"]

# Sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Datasets", "Testing"], 
        icons=['book', 'upload'], menu_icon="cast", default_index=0)
    selected

# ==============================
# MENU: DATASETS
# ==============================
if selected == 'Datasets':
    st.title("Datasets :")

    for category in categories:
        st.markdown(f"### {category.replace('-v1','').capitalize()}")

        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            st.warning(f"Folder '{category}' tidak ditemukan di {data_dir}")
            continue

        image_files = []
        for root, dirs, files in os.walk(category_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, f))

        if not image_files:
            st.info(f"Tidak ada gambar di folder {category}.")
        else:
            st.write(f"🖼️ Total gambar: **{len(image_files)}**")
            cols = st.columns(4)
            for i, img_path in enumerate(image_files[:12]):
                img_name = os.path.basename(img_path)
                image = Image.open(img_path)
                with cols[i % 4]:
                    st.image(image, caption=img_name, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Deteksi & Preprocessing Otomatis (MTCNN → Grayscale)")

    if st.button("Jalankan Deteksi + Preprocessing"):
        with st.spinner("Sedang mendeteksi wajah menggunakan MTCNN..."):
            mtcnn_faces, mtcnn_labels = show_all_data_mtcnn(data_dir, categories, max_per_cat=3)

        if mtcnn_faces:
            st.success(f"✅ Deteksi wajah selesai — total {len(mtcnn_faces)} wajah terdeteksi.")
            with st.spinner("Melakukan preprocessing (grayscale + normalisasi)..."):
                processed_images = preprocess_mtcnn_results(mtcnn_faces)
                show_preprocessed_samples(processed_images, mtcnn_labels, categories, n=10)
        else:
            st.warning("Tidak ada wajah yang terdeteksi pada dataset ini.")

# ==============================
# MENU: TESTING
# ==============================
elif selected == 'Testing':
    st.title("Testing")
    st.subheader("Testing Data Criminal vs Non-Criminal")

    model_path = "model/criminal_classifier.h5"

    # Upload gambar dari user
    uploaded_file = st.file_uploader("🖼️ Upload gambar wajah", type=["jpg", "jpeg", "png"])

    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.65, 0.05)

    if uploaded_file is not None:
        # Tampilkan preview gambar
        st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

        # Tombol untuk mulai prediksi
        if st.button("🔍 Mulai Prediksi"):
            with st.spinner("Sedang memproses dan mendeteksi wajah..."):
                result = predict_image_streamlit(
                    model_path=model_path,
                    img_file=uploaded_file,
                    categories=["criminal", "non-criminal"],
                    threshold=threshold
                )
            if result is None:
                st.warning("Tidak ada hasil prediksi yang valid.")
