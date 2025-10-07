import streamlit as st
import pandas as pd
import altair as alt
import requests
import time

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="AI Traffic Optimizer", layout="wide")
st.title("🚦 AI TRAFFIC MONITORING DASHBOARD")

# ------------------------------------------------------------
# API Config
# ------------------------------------------------------------
API_URL = "http://127.0.0.1:5000"
REFRESH_INTERVAL = 3 

def fetch_data():
    try:
        vehicle_data=requests.get(f"{API_URL}/vehicle-count").json()
        signal_status=requests.get(f"{API_URL}/signal-status").json()
    except:
        vehicle_data = {f"road{i}": {"count": 0 , "weight": 0}  for i in range(1, 5)}

        signal_status ={
            "cycle":{f"road{i}": {"status": "Red", "time": 0} for i in range(1, 5)}
        } 
    return vehicle_data,signal_status

# --- Video Files ---
video_files = [
    r"../backend/video_input/videos/t1.mp4",
    r"../backend/video_input/videos/t2.mp4",
    r"../backend/video_input/videos/t3.mp4",
    r"../backend/video_input/videos/t4.mp4"

]

# --- Placeholders ---
metrics_placeholders = {}
count_placeholders={}
weight_placeholders = {}
signal_placeholders = {}

road_names=["road1", "road2", "road3", "road4"]

cols = st.columns(2)
for idx, road in enumerate(road_names):
    with cols[idx % 2]:
        road_col, signal_col = st.columns([1, 2])
        with road_col:
            st.subheader(road.upper())  # e.g., ROAD1
        with signal_col:
            signal_placeholders[road] = st.empty()
            signal_placeholders[road].write("Signal: ⛔ Red | Time Left: 0 sec")

        st.video(video_files[idx], start_time=0)

        count_col, weight_col = st.columns(2)
        with count_col:
            count_placeholders[road] = st.empty()
            count_placeholders[road].metric("Vehicle Count", 0)
        with weight_col:
            weight_placeholders[road] = st.empty()
            weight_placeholders[road].metric("Vehicle Weight", 0)
        
st.markdown("---")
chart_placeholder = st.empty()

while True:
    vehicle_data, cycle_data = fetch_data()
    for road in road_names:
        data = vehicle_data.get(road, {"count": 0, "weight": 0})
        signal_info = cycle_data.get(road, {"status": "Red", "time": 0})

        count = data.get("count", 0)
        weight = data.get("weight", 0)
        status = signal_info.get("status", "Red")
        time_left = signal_info.get("time", 0)

        # Choose emoji color
        emoji = "🟢" if status == "Green" else "🔴"

        # Update display
        signal_placeholders[road].markdown(
            f"**{emoji} {status}** | ⏱️ Time Left: `{time_left} sec`",
            unsafe_allow_html=True
        )
        count_placeholders[road].metric("Vehicle Count", count)
        weight_placeholders[road].metric("Vehicle Weight", weight)

    # ------------------- Chart Update -------------------
    df = pd.DataFrame([
        {"Road": road,
         "Vehicle Count": vehicle_data.get(road, {}).get("count", 0),
         "Vehicle Weight": vehicle_data.get(road, {}).get("weight", 0)}
        for road in road_names
    ])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Road:N",
            y="Vehicle Weight:Q",
            color=alt.Color("Vehicle Count:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["Road", "Vehicle Count", "Vehicle Weight"]
        )
        .properties(height=400, title="🚗 Road-wise Vehicle Weight (Live)")
    )

    chart_placeholder.altair_chart(chart, use_container_width=True)
    time.sleep(REFRESH_INTERVAL)