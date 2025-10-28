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
from utils import show_all_data_mtcnn

# Direktori dataset
data_dir = "data"
categories = ["criminal-v1", "non-criminal-v1"]

# Sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Datasets", "Testing"], 
        icons=['house', 'gear', 'book'], menu_icon="cast", default_index=0)
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
    st.subheader("🔍 Hasil Deteksi Wajah (MTCNN)")
    if st.button("Tampilkan Deteksi MTCNN"):
        with st.spinner("Sedang memproses gambar..."):
            show_all_data_mtcnn(data_dir, categories, max_per_cat=3)
        st.success("Selesai menampilkan hasil deteksi wajah!")

# ==============================
# MENU: TESTING
# ==============================
elif selected == 'Testing':
    st.title("Testing")
    st.subheader("Test Data Detak Jantung")
