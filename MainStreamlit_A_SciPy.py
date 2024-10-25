import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model prediksi
model_kategori = pickle.load(open('BestModel_CLF_RF_SciPy.pkl', 'rb'))
model_harga = pickle.load(open('BestModel_Lasso_SciPy.pkl', 'rb'))

# Sidebar menu
with st.sidebar:
    selected = st.selectbox("Pilih Menu", ["Prediksi Kategori Properti", "Prediksi Harga Properti"])

# Fungsi untuk prediksi kategori properti
def prediksi_kategori(data):
    pred = model_kategori.predict(data)
    return pred[0]

# Fungsi untuk prediksi harga properti
def prediksi_harga(data):
    pred = model_harga.predict(data)
    return pred[0]

# Tampilan aplikasi berdasarkan menu yang dipilih
if selected == "Prediksi Kategori Properti":
    st.title("Prediksi Kategori Properti")
    
    # Input fitur (sesuaikan dengan model Anda)
    luas_bangunan = st.number_input("Luas Bangunan (m2)", min_value=0.0)
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0.0)
    jumlah_kamar = st.slider("Jumlah Kamar", 1, 10)
    lokasi = st.selectbox("Lokasi Properti", ["Jakarta", "Bandung", "Surabaya", "Yogyakarta"])
    
    # Preprocessing input (contoh, sesuaikan dengan kebutuhan model Anda)
    data = pd.DataFrame([[luas_bangunan, luas_tanah, jumlah_kamar, lokasi]], 
                        columns=['luas_bangunan', 'luas_tanah', 'jumlah_kamar', 'lokasi'])
    
    # Tombol prediksi
    if st.button("Prediksi Kategori"):
        hasil = prediksi_kategori(data)
        st.success(f"Kategori Properti: {hasil}")

elif selected == "Prediksi Harga Properti":
    st.title("Prediksi Harga Properti")
    
    # Input fitur (sesuaikan dengan model Anda)
    luas_bangunan = st.number_input("Luas Bangunan (m2)", min_value=0.0)
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0.0)
    jumlah_kamar = st.slider("Jumlah Kamar", 1, 10)
    lokasi = st.selectbox("Lokasi Properti", ["Jakarta", "Bandung", "Surabaya", "Yogyakarta"])
    
    # Preprocessing input (contoh, sesuaikan dengan kebutuhan model Anda)
    data = pd.DataFrame([[luas_bangunan, luas_tanah, jumlah_kamar, lokasi]], 
                        columns=['luas_bangunan', 'luas_tanah', 'jumlah_kamar', 'lokasi'])
    
    # Tombol prediksi
    if st.button("Prediksi Harga"):
        harga = prediksi_harga(data)
        st.success(f"Harga Properti: Rp {harga:,.0f}")

# Catatan: Sesuaikan input, fitur, dan preprocessing berdasarkan model yang Anda gunakan.
