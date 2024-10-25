import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

with open("BestModel_REG_Lasso_SciPy.pkl", "rb") as file:
    lr_model = pickle.load(file)

with open("BestModel_CLF_RF_SciPy.pkl", "rb") as file:
    rf_model = pickle.load(file)

with st.sidebar:
    selected = option_menu('Tutorial Desain Streamlit UTS ML 24/25',
                           ['Klasifikasi', 'Regresi'],
                           default_index=0)

if selected == 'Klasifikasi':
    st.title('Klasifikasi')
    
    st.write("Silakan upload file dataset Anda (format csv).")
    uploaded_file = st.file_uploader("Pilih file", type=["csv", "txt"])
    
    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        st.write("Dataset yang diunggah:")
        st.dataframe(dataset.head())

    st.write("Masukkan data yang diperlukan untuk prediksi klasifikasi:")
    feature = st.selectbox("Pilih Fitur Utama", ["Under", "Normal", "Over"])
    
    if st.button("Prediksi Klasifikasi"):
        pred_input = [[feature]]  
        prediction = lr_model.predict(pred_input)
        st.write(f"Hasil prediksi klasifikasi: {prediction}")

if selected == 'Regresi':
    st.title('Regresi')
    
    st.write("Masukkan data yang diperlukan untuk prediksi regresi:")
    ukuran = st.slider("Ukuran Properti (m²)", 0, 500)
    kamar_tidur = st.slider("Jumlah Kamar Tidur", 1, 10)
    kamar_mandi = st.slider("Jumlah Kamar Mandi", 1, 5)
    
    if st.button("Prediksi Harga Properti"):
        reg_input = [[ukuran, kamar_tidur, kamar_mandi]]
        price_prediction = rf_model.predict(reg_input)
        st.write(f"Prediksi harga properti: Rp {price_prediction[0]:,.2f}")
