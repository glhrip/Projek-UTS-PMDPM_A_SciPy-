import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Load dataset
dataset_path = '/mnt/data/Dataset UTS_Gasal 2425.csv'
data = pd.read_csv(dataset_path)

# Sidebar menu
with st.sidebar:
    selected = option_menu('Tutorial Desain Streamlit UTS ML 24/25',
                           ['Klasifikasi', 'Regresi', 'Catatan'],
                           default_index=0)

# Klasifikasi Section
if selected == 'Klasifikasi':
    st.title('Klasifikasi')
    st.write("### Preview of Dataset")
    st.write(data.head())

    # Upload file option
    st.write('Upload Dataset')
    file = st.file_uploader("Upload a dataset", type=["csv", "txt"])

    if file:
        uploaded_data = pd.read_csv(file)
        st.write(uploaded_data.head())

    # User Inputs
    st.write('User Input Parameters')
    age = st.slider("Age", 0, 100)
    gender = st.radio("Gender", ["Female", "Male"])
    num_rooms = st.selectbox("Number of Rooms", data["numberofrooms"].unique())
    square_meters = st.number_input("Square Meters", min_value=0)
    has_yard = st.radio("Yard", ["yes", "no"])
    
    # Prediction Button
    predict = st.button("Predict Classification")
    if predict:
        # Dummy Classification Example
        st.write(f"Classifying based on {num_rooms} rooms and {square_meters} square meters.")

# Regresi Section
if selected == 'Regresi':
    st.title('Regresi')
    st.write("### Preview of Dataset")
    st.write(data.head())

    # Upload file option
    st.write('Upload Dataset')
    file = st.file_uploader("Upload a dataset", type=["csv", "txt"])

    if file:
        uploaded_data = pd.read_csv(file)
        st.write(uploaded_data.head())

    # User Inputs for Regression
    st.write('User Input Parameters')
    num_rooms = st.slider("Number of Rooms", int(data['numberofrooms'].min()), int(data['numberofrooms'].max()))
    num_floors = st.slider("Number of Floors", int(data['floors'].min()), int(data['floors'].max()))
    is_new = st.radio("Is it newly built?", ["new", "old"])

    # Prediction Button
    predict = st.button("Predict Price")
    if predict:
        # Dummy Regression Example
        st.write(f"Predicting price for a {is_new} building with {num_rooms} rooms and {num_floors} floors.")
