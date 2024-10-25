import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

with open("BestModel_CLF_RF_SciPy.pkl", "rb") as file:
    rf_model = pickle.load(file)

with open("BestModel_REG_Lasso_SciPy.pkl", "rb") as file:
    lr_model = pickle.load(file)

data = pd.read_csv("Dataset UTS_Gasal 2425.csv")

with st.sidebar:
    selected = option_menu('Streamlit UTS ML 24/25',
                           ['Klasifikasi', 'Regresi', 'Catatan'],
                           default_index=0)

if selected == 'Klasifikasi':
    st.title('Klasifikasi Properti')

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

    if st.button("Prediksi Kategori"):
        try:
            kategori = rf_model.predict(data_input)[0]
            st.success(f"Kategori Properti: {kategori}")
        except ValueError as e:
            st.error(f"Error dalam prediksi: {e}")

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
