import streamlit as st
import pandas as pd
from joblib import load
from datetime import timedelta
from streamlit_folium import st_folium
import folium

st.title("Taxi Predictions")
st.markdown(
    "> Using state-of-the-art **AI**, we can accurately predict how long your next taxi drive in NYC will be, and how far it will take you. Just enter the details of your trip below, and we'll do the rest!"
)


st.write("### Select your starting point on the map below:")
map = folium.Map(location=[40.7, -74.0], zoom_start=12)
folium.LatLngPopup().add_to(map)
st_data = st_folium(map, width=1200, height=600)

if st_data["last_clicked"] is None:
    st.error("Please select a starting point for your journey.")
    st.stop()

start_latitude = st_data["last_clicked"]["lat"]
start_longitude = st_data["last_clicked"]["lng"]

time_model = load("pipeline_time.joblib")
distance_model = load("pipeline_distance.joblib")


dayperioddict = {
    "Afternoon": "afternoon",
    "Late Night": "lateNight",
    "Morning": "morning",
    "Evening": "evening",
    "Night": "night",
}

st.write("### Select the details of your trip below:")

day_period = dayperioddict[st.selectbox("Time of day", dayperioddict.keys())]
season = st.radio("Season", ["Winter", "Summer", "Spring"])
day_name = st.select_slider(
    "Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
if st.checkbox("Is it raining?"):
    rain = 1
else:
    rain = 0
if st.checkbox("Is it snowing?"):
    snow = 1
else:
    snow = 0
temperature = st.slider("Temperature", -10.0, 30.0, 0.1)


time = time_model.predict(
    pd.DataFrame(
        {
            "dayPeriod": [day_period],
            "season": [season],
            "dayName": [day_name],
            "rain": [rain],
            "snow": [snow],
            "temperature": [temperature],
            "startLatitude": [start_latitude],
            "startLongitude": [start_longitude],
        }
    )
)
dist = distance_model.predict(
    pd.DataFrame(
        {
            "dayPeriod": [day_period],
            "season": [season],
            "dayName": [day_name],
            "rain": [rain],
            "snow": [snow],
            "temperature": [temperature],
            "startLatitude": [start_latitude],
            "startLongitude": [start_longitude],
        }
    )
)

st.write("---")

st.write(f"### Average Trip Distance Predicted `{dist[0].round(3)}` km")

st.write(f"### Average Trip Time Predicted: `{timedelta(seconds=time[0].round(0))}`")

chart = [(time[0]/60).round(1), (dist[0]).round(3)]

st.bar_chart(chart)
