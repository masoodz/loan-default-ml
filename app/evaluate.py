import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('data/train.csv')
print("Raw data loaded")

df = df.dropna(subset=['Loan_Status'])
print("Dropped missing targets, remaining rows:", len(df))

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

expected_cols = pd.read_csv('models/feature_columns.csv', header=None).squeeze().tolist()

preprocessor = joblib.load('models/preprocessor.pkl')

X_processed = preprocessor.transform(X)
print(f"Eval: X_val shape before split = {X_processed.shape}")

_, X_val, _, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print(f"Eval: X_val shape = {X_val.shape}")

model = load_model('models/loan_model.h5')

y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.5).astype(int)

print("Classification Report:\n", classification_report(y_val, y_pred_binary))
