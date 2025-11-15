
import streamlit as st
import pandas as pd
import joblib
import os

# Load the pre-trained pipeline
@st.cache_resource
def load_pipeline():
    # In a real deployment, ensure pipeline.joblib is accessible.
    # For this example, we assume it's in the same directory or a known path.
    # If deployed on Streamlit Cloud, you'd typically upload this file along with app.py
    pipeline_path = 'pipeline.joblib'
    if not os.path.exists(pipeline_path):
        st.error(f"Model file not found at {pipeline_path}. Please ensure it's uploaded.")
        st.stop()
    return joblib.load(pipeline_path)

pipeline = load_pipeline()

st.title('Diamond Price Prediction App')
st.write('Enter the diamond characteristics to predict its price.')

# Input features from the user
carat = st.slider('Carat', 0.2, 5.0, 0.7)
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.slider('Depth', 43.0, 79.0, 61.5)
table = st.slider('Table', 43.0, 95.0, 55.0)
x = st.slider('Length (mm)', 0.0, 11.0, 5.7)
y = st.slider('Width (mm)', 0.0, 11.0, 5.7)
z = st.slider('Depth (mm)', 0.0, 7.0, 3.5)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'carat': carat,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z
}])

if st.button('Predict Price'):
    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    st.success(f'Predicted Diamond Price: ${prediction:,.2f}')

