import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu('Tutorial Desain Streamlit UTS ML 24/25',
                           ['Klasifikasi',
                            'Regresi', 'Catatan'],
                            default_index=0)

if selected == 'Klasifikasi':
    st.title('Klasifikasi')

    st.write('Untuk Inputan File dataset (csv) bisa menggunakan st.file_uploader')
    file = st.file_uploader("Masukkan File", type=["csv", "txt"])
    st.write('Untuk usia bisa menggunakan st.slider')
    Age = st.slider("Age", 0, 100)
    st.write('Untuk jenis kelamin bisa menggunakan st.radio')
    Sex = st.radio("Gender", ["Female", "Male"])
    st.write('Untuk beberapa kolom bisa menggunakan st.selectbox')
    nama_kolom = st.selectbox("Nama Kolom", ["Under", "Normal", "Over"])
    
    # Input untuk panjang dan lebar
    st.write('Untuk inputan manual bisa menggunakan st.number_input')
    panjang = st.number_input("Masukan Input", 0)
    lebar = st.number_input("Masukan Nilai Lebar", 0)

    # Input jawaban dari pengguna
    jawaban = st.number_input("Masukan Jawaban Anda", min_value=0)
    st.write('Tombol button (Menggunakan st.button)')
    
    # Tombol untuk menghitung luas
    hitung = st.button("Prediksi")

    if hitung:
        luas_benar = panjang * lebar
        st.write(f"Panjang: {panjang}, Lebar: {lebar}")

        if jawaban == luas_benar:
            st.success(f"Benar! Luas Persegi Panjang adalah {luas_benar}.")
        else:
            st.error(f"Salah! Luas Persegi Panjang yang benar adalah {luas_benar}.")

if selected == 'Regresi':
    st.title('Regresi')

    st.write('Untuk Inputan File dataset (csv) bisa menggunakan st.file_uploader')
    file = st.file_uploader("Masukkan File", type=["csv", "txt"])
    st.write('Untuk usia bisa menggunakan st.slider')
    Age = st.slider("Age", 0, 100)
    st.write('Untuk jenis kelamin bisa menggunakan st.radio')
    Sex = st.radio("Gender", ["Female", "Male"])
    st.write('Untuk beberapa kolom bisa menggunakan st.selectbox')
    nama_kolom = st.selectbox("Nama Kolom", ["Under", "Normal", "Over"])
    
    # Input untuk panjang dan lebar
    st.write('Untuk inputan manual bisa menggunakan st.number_input')
    panjang = st.number_input("Masukan Input", 0)
    lebar = st.number_input("Masukan Nilai Lebar", 0)
    
    alas = st.slider("Masukkan Nilai Alas", 0, 100)
    tinggi = st.slider("Masukkan Nilai Tinggi", 0, 100)
    st.write('Tombol button (Menggunakan st.button)')
    hitung = st.button("Prediksi")

    if hitung:
        luas = 0.5 * alas * tinggi
        st.write("Luas Segitiga Adalah", luas)

# Halaman ketentuan
if selected == 'Catatan':
    st.title('Catatan')
    st.write('''1. Untuk memunculkan sidebar agar tidak error ketika di run, silahkan install library streamlit option menu di terminal dengan perintah "pip install streamlit-option-menu".')
    st.write('2. Menu yang dibuat ada 2 yaitu Klasifikasi dan Regresi.''')
    st.write('3. Inputnya apa saja, sesuaikan dengan arsitektur code anda pada notebook.')
    st.write('4. Referensi desain streamlit dapat di akses pada https://streamlit.io/')
    st.write('5. Link streamlit design ini dapat di akses pada https://apputs-6qzfrvr4ufiyzhj84mrfkt7.streamlit.app/')
    st.write('''6. Library dan file requirements yang dibutuhkan untuk deploy online di github ada 5 yaitu streamlit, scikit-learn, pandas, numpy, streamlit-option-menu.''')