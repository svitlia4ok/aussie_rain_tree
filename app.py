import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scripts.aussie_rain_process_user_data import *

def predict(humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, windSpeed3pm, date):
    model = joblib.load('model/aussie_rain_thin.joblib')
    model_params = joblib.load('model/add_data.joblib')
    data = np.expand_dims(np.array([humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, date, windSpeed3pm]), axis=0)
    user_data = pd.DataFrame(data, columns = model_params['columns'])
    user_data_preprocessed = preprocess_new_data(user_data, model_params['scaler'])  
    predictions = model.predict(user_data_preprocessed)   
    return predictions[0]

st.title('Передбачення дощу на наступний день за параметрами попереднього дня')
st.markdown('Дана модель передбачає ймовірність опадів на наступний день, отримуючи на вхід такі параметри як вологість повітря, кількість опадів, атмосферний тиск та інші')
st.image('images/autumn.jpg')
st.header('Параметри погоди')
col1, col2 = st.columns(2)
st.write()
with col1:
    st.text('Характеристики о 3 p.m.')
    humidity3pm = st.slider('Вологість 3 p.m.', 0, 100)
    pressure3pm = st.slider('Атмосферний тиск 3 p.m.', 870, 1100)
    cloud3pm = st.slider('Хмарність 3 p.m.', 0, 10)
    windSpeed3pm = st.slider('Швидкість вітру 3 p.m.', 0, 90)

with col2:
    st.text('Інші властивості')
    rainfall = st.slider('Опади', 0, 400)
    sunshine = st.slider('Сонячне світло', 0, 15)
    windgustspeed = st.slider('Швидкість пориву вітру', 0, 140)
    date = st.date_input('Дата', format="YYYY-MM-DD")

if (st.button('Прогнозувати')):
    result = predict(humidity3pm, rainfall, sunshine, pressure3pm, cloud3pm, windgustspeed, windSpeed3pm, date)
    if result:
        st.write('Згідно з прогнозом, очікується дощ.')
    else:
        st.write('Згідно з прогнозом, дощ не очікується')