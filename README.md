# Iris Flower Classification (Task 1)

## Overview
This project builds and evaluates multiple machine learning classification models to predict Iris flower species (Setosa, Versicolor, Virginica) based on physical dimensions (sepal/petal length and width).

## Implemented Algorithms & Comparison
- **Logistic Regression:** ~96.67% Accuracy
- **k-Nearest Neighbors (k-NN, k=3):** ~96.67% Accuracy
- **Decision Tree Classifier:** ~93.33% Accuracy

## Saved Artifacts
- `iris_model.pkl`: Serialized trained model.
- `iris_scaler.pkl`: StandardScaler fitted on training features.

## How to Run Inference
```python
import joblib
import numpy as np

# Load trained pipeline components
model = joblib.load('iris_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Provide sample features: [sepal length, sepal width, petal length, petal width]
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
scaled_sample = scaler.transform(sample_data)

# Predict target class
species_names = ['setosa', 'versicolor', 'virginica']
prediction = model.predict(scaled_sample)[0]

print(f"Predicted Species: {species_names[prediction]}")