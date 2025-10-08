// dashboard.js
console.log("✅ Dashboard script loaded");

// ---------------- CONFIG ----------------
const BASE_URL = "http://127.0.0.1:5000"; // Flask backend
const VEHICLE_API = `${BASE_URL}/vehicle-count`;
const SIGNAL_API = `${BASE_URL}/signal-status`;
const REFRESH_INTERVAL = 3000; // refresh every 3 sec

// ---------------- DOM ELEMENTS ----------------
const roads = {
  "Road 1": { countEl: "count-lane1", weightEl: "weight-lane1", signalEl: "signal-text-north", timeEl: "time-left-north", videoEl: "video-lane1" },
  "Road 2": { countEl: "count-lane2", weightEl: "weight-lane2", signalEl: "signal-text-east", timeEl: "time-left-east", videoEl: "video-lane2" },
  "Road 3": { countEl: "count-lane3", weightEl: "weight-lane3", signalEl: "signal-text-south", timeEl: "time-left-south", videoEl: "video-lane3" },
  "Road 4": { countEl: "count-lane4", weightEl: "weight-lane4", signalEl: "signal-text-west", timeEl: "time-left-west", videoEl: "video-lane4" },
};

const statusMessage = document.getElementById("status-message");

// ---------------- UPDATE UI FUNCTIONS ----------------
function updateVehicleUI(data) {
  for (const roadName in roads) {
    const el = roads[roadName];
    const info = data[roadName];
    if (!info) continue;

    document.getElementById(el.countEl).textContent = info.count;
    document.getElementById(el.weightEl).textContent = info.weight;
  }
}

function updateSignalUI(data) {
  const cycle = data.cycle;

  for (const roadName in roads) {
    const el = roads[roadName];
    const info = cycle[roadName];
    if (!info) continue;

    const signalText = document.getElementById(el.signalEl);
    const timeText = document.getElementById(el.timeEl);
    const video = document.getElementById(el.videoEl);

    // Update signal text & class
    signalText.textContent = `Signal: ${info.status}`;
    signalText.className = info.status === "Green" ? "signal-green" : "signal-red";

    // Update timer
    timeText.textContent = `Time: ${info.time} sec`;

    // Ensure all videos are playing; visually highlight the green one
    if (video) {
      video.play().catch(() => {}); // try to play all videos
      video.style.filter = info.status === "Green" ? "brightness(1)" : "brightness(0.4)";
    }
  }
}

// ---------------- FETCH FUNCTIONS ----------------
async function fetchData() {
  try {
    const [vehicleRes, signalRes] = await Promise.all([
      fetch(VEHICLE_API),
      fetch(SIGNAL_API)
    ]);

    if (!vehicleRes.ok || !signalRes.ok) throw new Error("API response not OK");

    const vehicleData = await vehicleRes.json();
    const signalData = await signalRes.json();

    updateVehicleUI(vehicleData);
    updateSignalUI(signalData);

    // Hide warning if working
    statusMessage.classList.add("hidden");

    console.log("🔁 Updated UI:", { vehicleData, signalData });
  } catch (err) {
    console.error("❌ API Error:", err);
    statusMessage.classList.remove("hidden");
  }
}

// ---------------- INITIALIZE ----------------
document.addEventListener("DOMContentLoaded", () => {
  fetchData();
  setInterval(fetchData, REFRESH_INTERVAL);
});
