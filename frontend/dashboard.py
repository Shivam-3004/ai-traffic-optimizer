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
    """Fetch vehicle count and signal status from API. If fail, return default values."""
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
signal_placeholders = {}

# --- Layout: 2x2 Grid ---
cols = st.columns(2)
lane_names = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

for idx, lane in enumerate(lane_names):
    col = cols[idx % 2] if idx < 2 else cols[idx % 2].container()
    with col:
        st.subheader(lane)
        signal_placeholders[lane] = st.empty()
        st.video(video_files[idx])
        metrics_placeholders[lane] = st.empty()
        metrics_placeholders[lane].metric("Vehicle Count", 0, "Density: Low")
        signal_placeholders[lane].write("Signal: Red | Time Left: 0 sec")

# --- Altair Chart Placeholder ---
chart_placeholder = st.empty()

# --- Footer (Loop ke bahar) ---
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:center; padding:10px; font-size:14px; color:grey;">
    🚦 <b>AI Traffic Optimizer</b><br>
    Built with <b>YOLO</b> (Object Detection) | <b>Flask</b> (Backend API) | <b>Streamlit</b> (Dashboard) | <b>Python</b> 🐍 <br><br>
    © 2025 <b>Team XYZ</b> | Hackathon Project <br>
    🔗 <a href="https://github.com/your-username/ai-traffic-optimizer" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Real-time Metrics & Chart Update Loop ---
REFRESH_INTERVAL = 2  # seconds
while True:
    vehicle_data, signal_data = fetch_data()
    lane_data = {}

    for i, lane in enumerate(lane_names, 1):
        count = vehicle_data.get(f"lane{i}", 0)
        density = "High" if count > 10 else "Medium" if count > 5 else "Low"
        status = signal_data.get(f"lane{i}", {}).get("status", "Red")
        time_left = signal_data.get(f"lane{i}", {}).get("time", 0)

        lane_data[lane] = {"count": count, "density": density, "status": status, "time": time_left}

        # Update metrics and signals dynamically
        metrics_placeholders[lane].metric("Vehicle Count", count, f"Density: {density}")
        signal_placeholders[lane].write(f"Signal: {status} | Time Left: {time_left} sec")

    # --- Update chart dynamically ---
    df = pd.DataFrame([{"Lane": lane, "Vehicle Count": data["count"], "Density": data["density"]} 
                       for lane, data in lane_data.items()])

    density_colors = alt.Scale(domain=["High", "Medium", "Low", "Very Low"],
                               range=["red", "orange", "yellow", "green"])

    chart = (alt.Chart(df)
             .mark_bar()
             .encode(
                 x=alt.X("Lane:N", title="Traffic Lanes"),
                 y=alt.Y("Vehicle Count:Q", title="Number of Vehicles"),
                 color=alt.Color("Density:N", scale=density_colors, legend=alt.Legend(title="Vehicle Density")),
                 tooltip=["Lane", "Vehicle Count", "Density"]
             )
             .properties(height=400, width=700, title="Lane-wise Vehicle Count"))

    chart_placeholder.altair_chart(chart, use_container_width=True)
    time.sleep(REFRESH_INTERVAL)
