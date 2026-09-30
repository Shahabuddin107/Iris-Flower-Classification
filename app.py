import streamlit as st
import numpy as np
import joblib

# Load trained artifacts
model = joblib.load('iris_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Enter feature dimensions to predict the Iris species:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(features)
    pred_idx = model.predict(scaled_features)[0]
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"### Predicted Species: **{species[pred_idx]}**")
