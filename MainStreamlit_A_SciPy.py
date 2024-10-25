import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained models (pastikan file model tersedia di direktori)
with open("model_kategori.pkl", "rb") as file:
    kategori_model = pickle.load(file)

with open("model_harga.pkl", "rb") as file:
    harga_model = pickle.load(file)

st.sidebar.title("Sistem Prediksi Properti PT. Ayodya Property")
pilihan = st.sidebar.radio("Pilih Menu", ["Prediksi Kategori Properti", "Prediksi Harga Properti"])

def encode_input(yes_no_value):
    return 1 if yes_no_value == "Yes" else 0

# Halaman Prediksi Kategori Properti
if pilihan == "Prediksi Kategori Properti":
    st.title("Prediksi Kategori Properti")

    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0.0)
    jumlah_kamar = st.slider("Jumlah Kamar", 1, 10)
    haspool = st.selectbox("Ada Kolam Renang?", ["Yes", "No"])
    citycode = st.number_input("Kode Kota", min_value=0)
    floors = st.number_input("Jumlah Lantai", min_value=1)

    input_data = np.array([[luas_tanah, jumlah_kamar, encode_input(haspool), citycode, floors]]).reshape(1, -1)

    if st.button("Prediksi Kategori"):
        kategori = kategori_model.predict(input_data)[0]
        st.success(f"Kategori Properti: {kategori}")

# Halaman Prediksi Harga Properti
elif pilihan == "Prediksi Harga Properti":
    st.title("Prediksi Harga Properti")

    squaremeters = st.number_input("Luas Tanah (m²)", min_value=0.0)
    numberofrooms = st.number_input("Jumlah Kamar", min_value=1)
    hasyard = st.selectbox("Ada Halaman?", ["Yes", "No"])
    isnewbuilt = st.selectbox("Properti Baru?", ["Yes", "No"])
    attic = st.number_input("Luas Loteng (m²)", min_value=0.0)
    garage = st.number_input("Luas Garasi (m²)", min_value=0.0)

    input_data = np.array([[squaremeters, numberofrooms, encode_input(hasyard), 
                            encode_input(isnewbuilt), attic, garage]]).reshape(1, -1)

    if st.button("Prediksi Harga"):
        harga = harga_model.predict(input_data)[0]
        st.success(f"Estimasi Harga Properti: Rp {harga:,.2f}")
