import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# Select features
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X = df[features]

# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Save the preprocessing objects
joblib.dump(scaler, 'scaler.sav')
joblib.dump(pca, 'pca.sav')