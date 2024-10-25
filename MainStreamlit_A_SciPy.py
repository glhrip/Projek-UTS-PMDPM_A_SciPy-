import streamlit as st
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu

def load_model(model_file):
    model_path = os.path.join('BestModel_CLF_RF_SciPy.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

with st.sidebar:
    selected = option_menu('Pilih Analisis',
                           ['Analisis Kategori Properti', 'Analisis Harga Properti'],
                           default_index=0)

rf_model = load_model('BestModel_CLF_RF_SciPy.pkl')

if selected == 'Analisis Kategori Properti':
    st.title("Klasifikasi Jenis Properti")
    st.write("Masukkan fitur properti untuk mengetahui kategori properti.")

    squaremeters = st.number_input("Luas Properti (m2)", min_value=0)
    numberofrooms = st.number_input("Jumlah Kamar Dalam Properti", min_value=0)
    hasyard = st.selectbox("Apakah Properti Memiliki Halaman?", ["yes", "no"])
    haspool = st.selectbox("Apakah Properti Memiliki Kolam Renang?", ["yes", "no"])
    floors = st.number_input("Jumlah Lantai Pada Properti", min_value=0)
    citycode = st.number_input("Kode Kota Properti", min_value=0)
    citypartrange = st.number_input("Rentang Wilayah Properti", min_value=0)
    numprevowners = st.number_input("Jumlah Pemilik Properti Sebelumnya", min_value=0)
    made = st.number_input("Tahun Pembuatan Properti", min_value=1900, max_value=2024)
    isnewbuilt = st.selectbox("Apakah Properti Baru / Lama?", ["yes", "no"])
    hasstormprotector = st.selectbox("Apakah Properti Memiliki Pelindung Badai?", ["yes", "no"])
    basement = st.number_input("Luas Basement Properti (m2)", min_value=0)
    attic = st.number_input("Luas Loteng Properti (m2)", min_value=0)
    garage = st.number_input("Kapasitas Garasi", min_value=0)
    hasstorageroom = st.selectbox("Apakah Properti Memiliki Ruang Penyimpanan?", ["yes", "no"])
    hasguestroom = st.number_input("Berapa Jumlah Kamar Tamu Di Properti?", min_value=0)

    angkaBinary = {"yes": 1, "no": 0}
    input_hasyard = angkaBinary[hasyard]
    input_haspool = angkaBinary[haspool]
    input_isnewbuilt = angkaBinary[isnewbuilt]
    input_hasstormprotector = angkaBinary[hasstormprotector]
    input_hasstorageroom = angkaBinary[hasstorageroom]

    input_data = np.array([[squaremeters, numberofrooms, input_hasyard, input_haspool, floors, citycode, 
                            citypartrange, numprevowners, made, input_isnewbuilt, input_hasstormprotector, 
                            basement, attic, garage, input_hasstorageroom, hasguestroom]])

    if st.button("Lakukan Prediksi"):
        prediction = rf_model.predict(input_data)[0]
        kategori_map = {0: 'Basic', 1: 'Luxury', 2: 'Middle'}
        kategori_properti = kategori_map.get(prediction, "Tidak Diketahui")
        st.success(f"Kategori properti: *{kategori_properti}*")

if selected == 'Regresi':
    st.title('Regresi Harga Properti')

    squaremeters = st.slider("Squaremeters", 0, 100000)
    numberofrooms = st.slider("Number of Rooms", 0, 100)
    hasyard = st.radio("Has Yard?", ["Yes", "No"])
    haspool = st.radio("Has Pool?", ["Yes", "No"])
    floors = st.number_input("Floors", 0)
    citycode = st.number_input("City Code", 0)
    citypartrange = st.number_input("City Part Range", 0)
    numprevowners = st.number_input("Number of Previous Owners", 0)
    made = st.number_input("Year Built", 0)
    isnewbuilt = st.radio("Is New Built?", ["New", "Old"])
    hasstormprotector = st.radio("Has Storm Protector?", ["Yes", "No"])
    basement = st.number_input("Basement Area", 0)
    attic = st.number_input("Attic Area", 0)
    garage = st.number_input("Garage Area", 0)
    hasstorageroom = st.radio("Has Storage Room?", ["Yes", "No"])
    hasguestroom = st.number_input("Number of Guest Rooms", 0)

    data_input = pd.DataFrame([[squaremeters, numberofrooms, hasyard == "Yes", haspool == "Yes", floors,
                                citycode, citypartrange, numprevowners, made, isnewbuilt == "New",
                                hasstormprotector == "Yes", basement, attic, garage, 
                                hasstorageroom == "Yes", hasguestroom]],
                              columns=['squaremeters', 'numberofrooms', 'hasyard', 'haspool', 'floors',
                                       'citycode', 'citypartrange', 'numprevowners', 'made', 
                                       'isnewbuilt', 'hasstormprotector', 'basement', 'attic', 
                                       'garage', 'hasstorageroom', 'hasguestroom'])

    if st.button("Prediksi Harga"):
        try:
            harga = lr_model.predict(data_input)[0]
            st.success(f"Harga Properti: Rp {harga:,.0f}")
        except ValueError as e:
            st.error(f"Error dalam prediksi: {e}")

if selected == 'Catatan':
    st.title('Catatan')
    st.write('''
    1. Untuk memunculkan sidebar agar tidak error ketika di run, silahkan install library streamlit option menu di terminal dengan perintah "pip install streamlit-option-menu".
    2. Menu yang dibuat ada 2 yaitu Klasifikasi dan Regresi.
    3. Inputnya apa saja, sesuaikan dengan arsitektur code anda pada notebook.
    4. Referensi desain streamlit dapat di akses pada https://streamlit.io/
    5. Link streamlit design ini dapat di akses pada https://apputs-6qzfrvr4ufiyzhj84mrfkt7.streamlit.app/
    6. Library dan file requirements yang dibutuhkan untuk deploy online di github ada 5 yaitu streamlit, scikit-learn, pandas, numpy, streamlit-option-menu.
    ''')
