import os
import traceback
import joblib
import cloudpickle
import streamlit as st

MODEL_PATH = "pipeline.joblib"

@st.cache_resource
def load_pipeline():
    # 1) Make sure file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please upload pipeline.joblib to the repo root.")
        st.stop()

    # 2) Try joblib first, then cloudpickle, and show full traceback on failure
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e_job:
        try:
            with open(MODEL_PATH, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e_cloud:
            st.error("Failed to load the model. Full debug information shown below.")
            st.text("Primary (joblib) error:")
            st.text(traceback.format_exc())  # shows the error from cloudpickle attempt
            # Provide the first error as well
            st.text("If you want the original joblib error, see the variable 'joblib_error' below.")
            st.text(str(e_job))
            st.stop()
   

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

