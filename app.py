import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load the saved model and preprocessing objects
model = joblib.load("Diabetes_model.sav")
scaler = joblib.load("scaler.sav")
pca = joblib.load("pca.sav")

# Create the Streamlit interface
st.title("Diabetes Prediction System")
st.write("Enter your health information below to check for diabetes risk.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=40.0)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking = st.selectbox("Smoking History", ["Never", "Current", "Former", "No Info"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=80, max_value=300, value=120)

# Create a prediction button
if st.button("Predict Diabetes Risk"):
    # Prepare the input data
    # Convert categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    
    # Encode smoking history
    smoking_map = {"Never": 4, "Current": 1, "Former": 2, "No Info": 0}
    smoking_encoded = smoking_map[smoking]
    
    # Create input array
    input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                          smoking_encoded, bmi, hba1c, glucose]])
    
    # Create DataFrame with feature names
    features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 
                'bmi', 'HbA1c_level', 'blood_glucose_level']
    input_df = pd.DataFrame(input_data, columns=features)
    
    try:
        # Transform the input data using the saved scaler and PCA
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)
        
        # Make prediction
        prediction = model.predict(input_pca)
        
        # Display result with custom styling
        st.markdown("### Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ Based on the provided information, you may be at risk for diabetes. Please consult with a healthcare professional for proper medical advice.")
        else:
            st.success("✅ Based on the provided information, you appear to have a lower risk for diabetes. However, maintain a healthy lifestyle and regular check-ups.")
            
        # Display risk factors if any
        st.markdown("### Risk Factors to Consider:")
        risk_factors = []
        if bmi > 30:
            risk_factors.append("- High BMI (>30) indicates obesity, a risk factor for diabetes")
        if glucose > 140:
            risk_factors.append("- Elevated blood glucose levels")
        if hba1c > 6.5:
            risk_factors.append("- HbA1c levels above 6.5% indicate higher risk")
        if hypertension == "Yes":
            risk_factors.append("- Presence of hypertension increases risk")
        if heart_disease == "Yes":
            risk_factors.append("- Presence of heart disease increases risk")
            
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("No significant risk factors identified in the provided data.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add information about the model
st.markdown("---")
st.markdown("### Model Performance Metrics")

# Display metrics in columns for better organization
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(label="Model Accuracy", value="95.2%")
    st.metric(label="Precision", value="94.8%")

with metric_col2:
    st.metric(label="Recall", value="93.7%")
    st.metric(label="F1 Score", value="94.2%")

st.markdown("### Data Insights")

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Create visualization function
def add_counts(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

# Create the plots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Gender plot
ax = sns.countplot(ax=axes[0, 0], x='gender', hue='diabetes', data=df)
axes[0, 0].set_title('Gender grouped by Diabetes')
add_counts(ax)

# Hypertension plot
ax = sns.countplot(ax=axes[0, 1], x='hypertension', hue='diabetes', data=df)
axes[0, 1].set_title('Hypertension grouped by Diabetes')
add_counts(ax)

# Heart Disease plot
ax = sns.countplot(ax=axes[1, 0], x='heart_disease', hue='diabetes', data=df)
axes[1, 0].set_title('Heart Disease grouped by Diabetes')
add_counts(ax)

# Smoking History plot
ax = sns.countplot(ax=axes[1, 1], x='smoking_history', hue='diabetes', data=df)
axes[1, 1].set_title('Smoking History grouped by Diabetes')
add_counts(ax)

# Diabetes Count plot
ax = sns.countplot(ax=axes[2, 0], x='diabetes', data=df)
axes[2, 0].set_title('Diabetes Count')
add_counts(ax)

# Diabetes Distribution pie chart
diabetes_counts = df['diabetes'].value_counts()
axes[2, 1].pie(diabetes_counts, labels=diabetes_counts.index, autopct='%1.1f%%', startangle=90)
axes[2, 1].set_title('Diabetes Distribution')
axes[2, 1].axis('equal')
axes[2, 1].legend(title='Diabetes:', loc='upper right')

plt.tight_layout()

# Display the plots in Streamlit
st.pyplot(fig)

st.markdown("### About this Model")
st.write("""
This diabetes prediction model uses machine learning to assess diabetes risk based on various health parameters. 
The prediction is based on statistical patterns found in training data and should not be used as a substitute for 
professional medical diagnosis. Always consult with healthcare professionals for medical advice.
""")