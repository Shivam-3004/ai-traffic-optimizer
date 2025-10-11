// dashboard.js
console.log("✅ Dashboard script loaded");

// ---------------- CONFIG ----------------
const BASE_URL = "http://127.0.0.1:5000"; // Flask backend
const VEHICLE_API = `${BASE_URL}/vehicle-count`;
const SIGNAL_API = `${BASE_URL}/signal-status`;
const REFRESH_INTERVAL = 3000; // refresh every 3 sec
const OVERLAY_DETECT_INTERVAL = 350; // ms, should match server DETECT_INTERVAL roughly

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
    const media = document.getElementById(el.videoEl);

    // Update signal text & class
    signalText.textContent = `Signal: ${info.status}`;
    signalText.className = info.status === "Green" ? "signal-green" : "signal-red";

    // Update timer
    timeText.textContent = `Time: ${info.time} sec`;

    // If the element is a video, try to play it. For images (raw-stream) we
    // cannot call play(); just adjust styling to highlight the green one.
    if (media) {
      try {
        if (media.tagName && media.tagName.toLowerCase() === 'video') {
          media.play().catch(() => {});
        }
      } catch (e) {}
      media.style.filter = info.status === "Green" ? "brightness(1)" : "brightness(0.4)";
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
  // start overlay polling for road4
  startOverlayPolling('road4');
});


// ---------------- OVERLAY (Road 4) ----------------
let _overlayTimer = null;
function startOverlayPolling(road) {
  // convert 'road4' -> 'lane4' to match element ids used in the DOM
  const laneId = road.replace(/^road/, 'lane');
  const canvas = document.getElementById(`overlay-${laneId}`);
  const img = document.getElementById(`video-${laneId}`);
  if (!canvas || !img) return;

  const ctx = canvas.getContext('2d');

  function resizeCanvas() {
    // set internal pixel buffer to match displayed size
    const w = img.clientWidth || img.width || 320;
    const h = img.clientHeight || img.height || 240;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }

  // resize once image is loaded (useful for initial layout)
  try {
    img.addEventListener('load', resizeCanvas);
  } catch (e) {}

  function drawBoxes(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data || !data.boxes) return;

    // server boxes are in pixels relative to source frame size; scale them to displayed size
    const sw = data.width || canvas.width;
    const sh = data.height || canvas.height;
    const sx = canvas.width / (sw || canvas.width);
    const sy = canvas.height / (sh || canvas.height);

    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.font = '14px Arial';
    ctx.fillStyle = 'lime';

    for (const b of data.boxes) {
      const x = b.x * sx;
      const y = b.y * sy;
      const w = b.w * sx;
      const h = b.h * sy;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(b.class || '', x + 4, y + 14);
    }
  }

  function poll() {
    // ensure canvas matches current displayed size
    if (!canvas.width || !canvas.height) resizeCanvas();
    fetch(`${BASE_URL}/detection-boxes/${road}`)
      .then(r => r.json())
      .then(drawBoxes)
      .catch(() => {});
  }

  // start interval
  _overlayTimer = setInterval(poll, OVERLAY_DETECT_INTERVAL);
  // initial poll
  poll();
  // resize when window changes
  window.addEventListener('resize', resizeCanvas);
}
