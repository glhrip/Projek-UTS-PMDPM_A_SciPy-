import streamlit as st
import pandas as pd
import pickle

# Load prediction models
with open('BestModel_CLF_RF_SciPy.pkl', 'rb') as file:
    model_kategori = pickle.load(file)
with open('BestModel_REG_Lasso_SciPy.pkl', 'rb') as file:
    model_harga = pickle.load(file)

# Sidebar menu
with st.sidebar:
    selected = st.selectbox("Pilih Menu", ["Prediksi Kategori Properti", "Prediksi Harga Properti"])

# Function to predict property category
def prediksi_kategori(data):
    pred = model_kategori.predict(data)
    return pred[0]

# Function to predict property price
def prediksi_harga(data):
    pred = model_harga.predict(data)
    return pred[0]

# Display application based on selected menu
if selected == "Prediksi Kategori Properti":
    st.title("Prediksi Kategori Properti")
    
    # Input features
    luas_bangunan = st.number_input("Luas Bangunan (m2)", min_value=0.0)
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0.0)
    jumlah_kamar = st.slider("Jumlah Kamar", 1, 10)
    lokasi = st.selectbox("Lokasi Properti", ["Jakarta", "Bandung", "Surabaya", "Yogyakarta"])
    
    # Preprocess input (adjust according to model requirements)
    data = pd.DataFrame([[luas_bangunan, luas_tanah, jumlah_kamar, lokasi]], 
                        columns=['luas_bangunan', 'luas_tanah', 'jumlah_kamar', 'lokasi'])
    
    # Predict button
    if st.button("Prediksi Kategori"):
        hasil = prediksi_kategori(data)
        st.success(f"Kategori Properti: {hasil}")

elif selected == "Prediksi Harga Properti":
    st.title("Prediksi Harga Properti")
    
    # Input features
    luas_bangunan = st.number_input("Luas Bangunan (m2)", min_value=0.0)
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0.0)
    jumlah_kamar = st.slider("Jumlah Kamar", 1, 10)
    lokasi = st.selectbox("Lokasi Properti", ["Jakarta", "Bandung", "Surabaya", "Yogyakarta"])
    
    # Preprocess input (adjust according to model requirements)
    data = pd.DataFrame([[luas_bangunan, luas_tanah, jumlah_kamar, lokasi]], 
                        columns=['luas_bangunan', 'luas_tanah', 'jumlah_kamar', 'lokasi'])
    
    # Predict button
    if st.button("Prediksi Harga"):
        harga = prediksi_harga(data)
        st.success(f"Harga Properti: Rp {harga:,.0f}")
