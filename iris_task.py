import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Dataset Load
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=== DATASET PREVIEW ===")
print(df.head())
print("\nMissing values:", df.isnull().sum().to_dict())

# 2. EDA Plots
sns.pairplot(df, hue='species_name', markers=["o", "s", "D"])
plt.suptitle("Iris Feature Distribution & Separability", y=1.02)
plt.show()

plt.figure(figsize=(7, 4))
numeric_df = df.drop(columns=['species_name'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. Train-Test Split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "k-NN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

best_model = None
best_acc = 0.0
best_name = ""

for name, model in models.items():
    if name in ["k-NN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    print(f"\n================ {name} ================")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=iris.target_names))
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# 5. Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.show()

# 6. Save Model Artifacts
joblib.dump(best_model, 'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
print("\nArtifacts saved: iris_model.pkl & iris_scaler.pkl")

# 7. Inference Demo
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
pred_idx = best_model.predict(sample_scaled)[0]
print(f"Sample Input: {sample[0]}")
print(f"Predicted Class: {iris.target_names[pred_idx]}")