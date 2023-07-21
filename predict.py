import streamlit as st
import pandas as pd
from joblib import load
from datetime import timedelta

time_model = load("pipeline_time.joblib")
distance_model = load("pipeline_distance.joblib")

st.title("Taxi Predictions")
st.markdown("> Using state-of-the-art *AI*, we can accurately predict how long your next taxi drive in NYC will be, and how far it will take you. Just enter the details of your trip below, and we'll do the rest!")

day_period = st.selectbox("Time of day", ['afternoon', 'lateNight', 'morning', 'evening', 'night'])
season = st.selectbox("Season", ['Winter', 'Summer', 'Spring'])
day_name = st.selectbox("Day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
if st.checkbox("Is it raining?"):
    rain = 1
else:
    rain = 0
if st.checkbox("Is it snowing?"):
    snow = 1
else:
    snow = 0
temperature = st.slider("Temperature", -10.0, 30.0, 0.1)
start_latitude = float(st.text_input("Start latitude", 40.7))
start_longitude = float(st.text_input("Start Longitude", -74.0))

time = time_model.predict(pd.DataFrame({"dayPeriod": [day_period], "season": [season], "dayName": [day_name], "rain": [rain], "snow": [snow], "temperature": [temperature], "startLatitude": [start_latitude], "startLongitude": [start_longitude]}))
dist = distance_model.predict(pd.DataFrame({"dayPeriod": [day_period], "season": [season], "dayName": [day_name], "rain": [rain], "snow": [snow], "temperature": [temperature], "startLatitude": [start_latitude], "startLongitude": [start_longitude]}))

st.write(f"### Distance Prediction: `{dist[0].round(0)}` km")

st.write(f"### Time Prediction: `{timedelta(seconds=time[0].round(0))}`")
