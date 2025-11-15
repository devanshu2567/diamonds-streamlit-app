import streamlit as st
import pandas as pd
import joblib
import os
import traceback

st.set_page_config(page_title="Diamond Price Prediction", layout="centered")

st.title('Diamond Price Prediction App')
st.write('Enter the diamond characteristics to predict its price.')

# safer loader: returns pipeline or None and shows errors
@st.cache_resource
def load_pipeline_safe(path="pipeline.joblib"):
    if not os.path.exists(path):
        return None, f"Model file not found at {path}. Please upload pipeline.joblib to the repo root."
    try:
        pipeline = joblib.load(path)
        return pipeline, None
    except Exception as e:
        # try cloudpickle if joblib fails (optional)
        try:
            import cloudpickle
            with open(path, "rb") as f:
                pipeline = cloudpickle.load(f)
            return pipeline, None
        except Exception:
            tb = traceback.format_exc()
            return None, f"Failed to load model. Error:\n{tb}"

pipeline, load_error = load_pipeline_safe()

if load_error:
    st.warning(load_error)
    st.info("The UI will still let you build an input; prediction will be disabled until the model loads successfully.")

# Input features from the user (defaults set for safe types)
carat = st.slider('Carat', 0.2, 5.0, 0.7)
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.slider('Depth', 43.0, 79.0, 61.5)
table = st.slider('Table', 43.0, 95.0, 55.0)
x = st.slider('Length (mm)', 0.0, 11.0, 5.7)
y = st.slider('Width (mm)', 0.0, 11.0, 5.7)
z = st.slider('Depth (mm)', 0.0, 7.0, 3.5)

st.markdown("---")
st.write("Preview of the input that will be passed to the model:")

# Build the DataFrame inside a try/except to avoid NameError
try:
    input_dict = {
        'carat': float(carat),
        'cut': str(cut),
        'color': str(color),
        'clarity': str(clarity),
        'depth': float(depth),
        'table': float(table),
        'x': float(x),
        'y': float(y),
        'z': float(z)
    }
    input_data = pd.DataFrame([input_dict])
    st.dataframe(input_data)
except Exception as e:
    st.error("Error preparing input DataFrame. Details:")
    st.text(traceback.format_exc())
    input_data = None

if st.button('Predict Price'):
    # Guard: only attempt prediction if model loaded and input_data is valid
    if pipeline is None:
        st.error("Prediction unavailable because model failed to load. See the warning at the top for details.")
    elif input_data is None:
        st.error("Prediction unavailable because input data could not be created.")
    else:
        try:
            # Make prediction and format output
            prediction = pipeline.predict(input_data)[0]
            st.success(f'Predicted Diamond Price: ${prediction:,.2f}')
        except Exception:
            st.error("Model prediction failed. See full exception below:")
            st.text(traceback.format_exc())
