import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# Load dataset
df = pd.read_csv('diabetes.csv')

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Model: Linear SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"✅ Training Accuracy: {train_acc}")
print(f"✅ Testing Accuracy: {test_acc}")

# Classification Report
print("\n📊 Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save for later display
plt.close()

# Cross-Validation Score (optional)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"📌 Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Save model & scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved as 'model.pkl' and 'scaler.pkl'")
