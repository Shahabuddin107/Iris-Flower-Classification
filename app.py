import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

@st.cache_resource
def load_and_train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)
    
    return model, scaler, iris.target_names

# Model automatically ready ho jayega
model, scaler, target_names = load_and_train_model()

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
    
    result = target_names[pred_idx].capitalize()
    st.success(f"### Predicted Species: **{result}**")
