import streamlit as st
import pandas as pd
import altair as alt
import requests
import time

# --- Page Config ---
st.set_page_config(page_title="AI Traffic Analyzer", layout="wide")
st.title("🚦 AI TRAFFIC MONITORING DASHBOARD")

# --- API Config ---
API_URL = "http://127.0.0.1:5000"

def fetch_data():
    try:
        vehicle_data = requests.get(f"{API_URL}/vehicle-count").json()
        signal_data = requests.get(f"{API_URL}/signal-status").json()
    except:
        vehicle_data = {f"lane{i}": 0 for i in range(1, 5)}
        signal_data = {f"lane{i}": {"status": "Red", "time": 0} for i in range(1, 5)}
    return vehicle_data, signal_data

# --- Video Files ---
video_files = [
    "3285790-hd_1920_1080_30fps.mp4",
    "3285790-hd_1920_1080_30fps.mp4",
    "3285790-hd_1920_1080_30fps.mp4",
    "3285790-hd_1920_1080_30fps.mp4"
]

# --- Placeholders ---
metrics_placeholders = {}
density_placeholders = {}
signal_placeholders = {}

lane_names = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

# --- 2x2 Grid ---
cols = st.columns(2)

for idx, lane in enumerate(lane_names):
    col = cols[idx % 2] if idx < 2 else cols[idx % 2].container()
    with col:
        # --- Lane name and signal in same row ---
        lane_col, signal_col = st.columns([1, 2])
        with lane_col:
            st.subheader(lane)
        with signal_col:
            signal_placeholders[lane] = st.empty()
            signal_placeholders[lane].write("Signal: Red | Time Left: 0 sec")
        
        # --- Video below ---
        st.video(video_files[idx], start_time=0, format="video/mp4", width=400)
        # --- Vehicle count & density in same row (below video) ---
        count_col, density_col = st.columns(2)
        st.markdown("<hr>", unsafe_allow_html=True)
        with count_col:
            metrics_placeholders[lane] = st.empty()
            metrics_placeholders[lane].metric("Vehicle Count", 0)
        with density_col:
            density_placeholders[lane] = st.empty()
            density_placeholders[lane].metric("Density", "Low")
       


st.markdown("---")

# --- Altair Chart Placeholder ---
chart_placeholder = st.empty()

# --- Footer ---
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:center; padding:10px; font-size:14px; color:grey;">
    🚦 <b>AI Traffic Optimizer</b><br>
    Built with <b>YOLO</b> | <b>Flask</b> | <b>Streamlit</b> | <b>Python</b> 🐍 <br><br>
    © 2025 <b>Team XYZ</b> | Hackathon Project <br>
    🔗 <a href="https://github.com/your-username/ai-traffic-optimizer" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Real-time Loop ---
REFRESH_INTERVAL = 2
while True:
    vehicle_data, signal_data = fetch_data()
    for i, lane in enumerate(lane_names, 1):
        count = vehicle_data.get(f"lane{i}", 0)
        density = "High" if count > 10 else "Medium" if count > 5 else "Low"
        status = signal_data.get(f"lane{i}", {}).get("status", "Red")
        time_left = signal_data.get(f"lane{i}", {}).get("time", 0)

        # Update placeholders
        signal_placeholders[lane].write(f"Signal: {status} | Time Left: {time_left} sec")
        metrics_placeholders[lane].metric("Vehicle Count", count)
        density_placeholders[lane].metric("Density", density)

    # --- Update chart dynamically ---
    df = pd.DataFrame([
        {"Lane": lane,
         "Vehicle Count": vehicle_data.get(f"lane{i}", 0),
         "Density": "High" if vehicle_data.get(f"lane{i}", 0) > 10 else "Medium" if vehicle_data.get(f"lane{i}", 0) > 5 else "Low"}
        for i, lane in enumerate(lane_names, 1)
    ])

    chart = alt.Chart(df).mark_bar().encode(
        x="Lane:N",
        y="Vehicle Count:Q",
        color=alt.Color("Density:N", scale=alt.Scale(domain=["High","Medium","Low"], range=["red","orange","green"])),
        tooltip=["Lane", "Vehicle Count", "Density"]
    ).properties(height=400, width=700, title="Lane-wise Vehicle Count")

    chart_placeholder.altair_chart(chart, use_container_width=True)
    time.sleep(REFRESH_INTERVAL)
