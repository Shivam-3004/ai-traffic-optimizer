import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="AI Traffic Analyzer", layout="wide")
st.title("🚦 AI Traffic Analyzer")

lane_data = {
    "Lane 1": {"count": 12, "density": "High"},
    "Lane 2": {"count": 8, "density": "Medium"},
    "Lane 3": {"count": 5, "density": "Low"},
    "Lane 4": {"count": 15, "density": "High"}
}

df = pd.DataFrame([
    {"Lane": lane, "Vehicle Count": data["count"], "Density": data["density"]}
    for lane, data in lane_data.items()
])


st.markdown("""
    <style>
    video {
        height: 250px !important;
        width: 100% !important;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Lane 1")

    st.video(r"F:\Drive C content\hackathon\ai-traffic-optimizer\frontend\second\3285790-hd_1920_1080_30fps.mp4" )
    m1, m2 = st.columns(2)
    m1.metric("vehicle_count :",lane_data["Lane 1"]["count"])
    m2.metric("vehicle_density :",lane_data["Lane 1"]["density"])
with col2:
    st.subheader("Lane 2")
    st.video(r"F:\Drive C content\hackathon\ai-traffic-optimizer\frontend\second\3285790-hd_1920_1080_30fps.mp4" )
    m1, m2 = st.columns(2)
    m1.metric("vehicle_count :",lane_data["Lane 2"]["count"])
    m2.metric("vehicle_density :",lane_data["Lane 2"]["density"])

col3, col4 = st.columns(2)
with col3:
    st.subheader("Lane 3")
    st.video(r"F:\Drive C content\hackathon\ai-traffic-optimizer\frontend\second\3285790-hd_1920_1080_30fps.mp4" )
    m1, m2 = st.columns(2)
    m1.metric("vehicle_count :",lane_data["Lane 3"]["count"])
    m2.metric("vehicle_density :",lane_data["Lane 3"]["density"])

with col4:
    st.subheader("Lane 4")
    st.video(r"F:\Drive C content\hackathon\ai-traffic-optimizer\frontend\second\3285790-hd_1920_1080_30fps.mp4" )
    m1, m2 = st.columns(2)
    m1.metric("vehicle_count :",lane_data["Lane 4"]["count"])
    m2.metric("vehicle_density :",lane_data["Lane 4"]["density"])

st.markdown("## 📊 Lane-wise Vehicle Count with Density Colors")
density_colors = alt.Scale(
domain=["High", "Medium", "Low", "Very Low"],
range=["red", "orange", "yellow", "green"]
)

chart = (
alt.Chart(df)
.mark_bar()
.encode(
    x=alt.X("Lane:N", title="Traffic Lanes"),
    y=alt.Y("Vehicle Count:Q", title="Number of Vehicles"),
    color=alt.Color("Density:N", scale=density_colors, legend=alt.Legend(title="Vehicle Density")),
    tooltip=["Lane", "Vehicle Count", "Density"]
)
.properties(height=400, width=700, title="Lane-wise Vehicle Count")
)

st.altair_chart(chart, use_container_width=True)
