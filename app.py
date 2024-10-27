import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scripts.aussie_rain_process_user_data import *

# Prediction function using a pre-trained model
def predict(humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, windSpeed3pm, date):
    # Load the pre-trained model and preprocessing parameters
    model = joblib.load('model/aussie_rain_thin.joblib')
    model_params = joblib.load('model/add_data.joblib')
    
    # Organize the user inputs into a NumPy array, then expand dimensions for model input compatibility
    data = np.expand_dims(np.array([humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, date, windSpeed3pm]), axis=0)
    
    # Convert data into a DataFrame using columns from the model's parameter dictionary
    user_data = pd.DataFrame(data, columns=model_params['columns'])
    
    # Preprocess the data (scaling, encoding, etc.) before prediction
    user_data_preprocessed = preprocess_new_data(user_data, model_params['scaler'])
    
    # Get prediction from the model
    predictions = model.predict(user_data_preprocessed)
    
    # Return the prediction result (1 = rain, 0 = no rain)
    return predictions[0]

# Streamlit UI setup
st.title('Next Day Rain Prediction Based on Previous Day Parameters')
st.markdown('This model predicts the likelihood of rain on the following day based on parameters such as humidity, rainfall, air pressure, and more.')
st.image('images/autumn.jpg')
st.header('Weather Parameters')

# Two-column layout for parameter input
col1, col2 = st.columns(2)

# Inputs for weather characteristics at 3 p.m. in the first column
with col1:
    st.text('Parameters at 3 p.m.')
    humidity3pm = st.slider('Humidity at 3 p.m.', 0, 100)
    pressure3pm = st.slider('Air Pressure at 3 p.m.', 870, 1100)
    cloud3pm = st.slider('Cloudiness at 3 p.m.', 0, 10)
    windSpeed3pm = st.slider('Wind Speed at 3 p.m.', 0, 90)

# Inputs for other weather features in the second column
with col2:
    st.text('Additional Parameters')
    rainfall = st.slider('Rainfall', 0, 400)
    sunshine = st.slider('Sunshine', 0, 15)
    windgustspeed = st.slider('Wind Gust Speed', 0, 140)
    date = st.date_input('Date', format="YYYY-MM-DD")

# Prediction trigger and output display
if st.button('Predict'):
    result = predict(humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, windSpeed3pm, date)
    if result:
        st.write('According to the forecast, rain is expected.')
    else:
        st.write('According to the forecast, no rain is expected.')