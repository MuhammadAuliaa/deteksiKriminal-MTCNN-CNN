import os
import random
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import streamlit as st

# Inisialisasi detector global biar gak buat ulang tiap kali
detector = MTCNN()

def show_all_data_mtcnn(data_dir, categories, max_per_cat=5):
    """
    Menampilkan hasil deteksi wajah dari beberapa kategori menggunakan MTCNN.
    Hasil langsung ditampilkan di Streamlit.
    """
    all_images = []
    labels = []

    for category in categories:
        folder = os.path.join(data_dir, category)
        img_files = []

        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_files.append(os.path.join(root, f))

        if not img_files:
            continue

        img_files = random.sample(img_files, min(max_per_cat, len(img_files)))
        all_images.extend(img_files)
        labels.extend([category] * len(img_files))

    if not all_images:
        st.warning("Tidak ada gambar ditemukan untuk dideteksi.")
        return

    # Plot hasil deteksi
    fig, axes = plt.subplots(1, len(all_images), figsize=(20, 5))
    if len(all_images) == 1:
        axes = [axes]  # supaya tetap iterable

    for idx, (img_path, label) in enumerate(zip(all_images, labels)):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)

        for res in results:
            x, y, w, h = res['box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            confidence = res['confidence']
            cv2.putText(image, f"{confidence:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        axes[idx].imshow(image)
        axes[idx].set_title(f"{label}\nFaces: {len(results)}")
        axes[idx].axis("off")

    plt.suptitle("All Categories - MTCNN Results", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)