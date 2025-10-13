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
  // overlay polling is started only when camera live is enabled; initialize UI
  const toggle = document.getElementById('toggle-road4');
  const img = document.getElementById('video-lane4');
  const fileVid = document.getElementById('video-lane4-file');
  const camSelect = document.getElementById('camera-index-road4');
  const camHelp = document.getElementById('camera-index-help');
  // populate camera index select from server
  fetch(`${BASE_URL}/list-cameras`).then(r => r.json()).then((data) => {
    if (!data || !data.results) return;
    data.results.forEach(entry => {
      const opt = document.createElement('option');
      opt.value = entry.index;
      opt.textContent = `${entry.index} - ${entry.opened ? (entry.read ? 'available' : 'no-frame') : 'not-opened'}`;
      camSelect.appendChild(opt);
    });
  }).catch(() => { camHelp.textContent = 'Unable to list cameras'; });
  if (toggle) {
    // Ensure initial visibility state
    if (fileVid) { fileVid.style.display = 'block'; fileVid.play().catch(()=>{}); }
    if (img) { img.classList.add('hidden'); img.src = ''; }

    // helper to show/hide loading spinner
    const showLoading = (show) => {
      const loader = document.getElementById('stream-loading-lane4');
      if (!loader) return;
      if (show) loader.classList.remove('hidden'); else loader.classList.add('hidden');
    }

    // snapshot fallback timer (poll /snapshot if MJPEG doesn't start)
    let _snapshotTimer = null;
    let _mjpegTimeout = null;
    const MJPEG_START_TIMEOUT = 2500; // ms to wait for mjpeg image to load
    const SNAPSHOT_POLL_INTERVAL = 1200; // ms between snapshot polls when fallback engaged

    function startSnapshotFallback() {
      stopSnapshotFallback();
      _snapshotTimer = setInterval(async () => {
        try {
          const resp = await fetch(`${BASE_URL}/snapshot/road4`);
          if (!resp.ok) return;
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          // show snapshot in the img element
          img.src = url;
          // revoke object URL after some time to avoid leaks
          setTimeout(() => { try { URL.revokeObjectURL(url); } catch(e){} }, 5000);
        } catch (e) {
          // ignore
        }
      }, SNAPSHOT_POLL_INTERVAL);
    }
    function stopSnapshotFallback() {
      if (_snapshotTimer) { clearInterval(_snapshotTimer); _snapshotTimer = null; }
    }

    // when the img receives its first natural frame, stop the fallback and hide loader
    if (img) {
      img.addEventListener('load', () => {
        // naturalWidth is >0 when actual image data arrived
        if (img.naturalWidth && img.naturalWidth > 0) {
          img.classList.remove('hidden');
          showLoading(false);
          stopSnapshotFallback();
          if (_mjpegTimeout) { clearTimeout(_mjpegTimeout); _mjpegTimeout = null; }
        }
      });
    }

    toggle.addEventListener('change', (ev) => {
      const on = ev.target.checked;
      if (on) {
        // ask server to switch source to camera first, include selected index
        const idx = camSelect ? camSelect.value : undefined;
        const url = idx ? `${BASE_URL}/switch-source?road=road4&mode=camera&index=${idx}` : `${BASE_URL}/switch-source?road=road4&mode=camera`;
        camHelp.textContent = 'Switching to camera...';
        showLoading(true);
        fetch(url, { method: 'GET' })
          .then(r => r.json())
          .then((res) => {
            if (res.ok) {
              // hide and pause file video to ensure only camera runs
              if (fileVid) { try { fileVid.pause(); } catch(e){}; fileVid.style.display = 'none'; }
              // set the MJPEG src; the image will become visible once frames arrive
              img.src = `${BASE_URL}/raw-stream/road4`;
              img.classList.remove('hidden');
              // start overlay polling and a timeout for mjpeg start
              startOverlayPolling('road4');
              camHelp.textContent = 'Camera live';
              // if no frame arrives within MJPEG_START_TIMEOUT, start snapshot fallback
              _mjpegTimeout = setTimeout(() => {
                // if img hasn't received a naturalWidth yet, start fallback
                if (!img.naturalWidth || img.naturalWidth === 0) {
                  startSnapshotFallback();
                  camHelp.textContent = 'Using snapshot fallback';
                }
                showLoading(false);
              }, MJPEG_START_TIMEOUT);
            } else {
              console.error('switch-source failed', res);
              toggle.checked = false;
              showLoading(false);
            }
          }).catch((e) => { console.error(e); toggle.checked = false; showLoading(false); });
      } else {
        // switch back to file: ask server to create file-based manager and release camera
        camHelp.textContent = 'Switching to file...';
        showLoading(true);
        fetch(`${BASE_URL}/switch-source?road=road4&mode=file`, { method: 'GET' })
          .then(r => r.json())
          .then((res) => {
            if (res.ok) {
              // stop camera image polling and hide it
              stopOverlayPolling();
              stopSnapshotFallback();
              if (img) { img.src = ''; img.classList.add('hidden'); }
              // show and play file video
              if (fileVid) { fileVid.style.display = 'block'; try { fileVid.play().catch(()=>{}); } catch(e){} }
              camHelp.textContent = 'File playback';
            } else {
              console.error('switch-source failed', res);
              toggle.checked = true;
            }
            showLoading(false);
          }).catch((e) => { console.error(e); toggle.checked = true; showLoading(false); });
      }
    });
    // default: leave file video visible; toggle off
    toggle.checked = false;
  }
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

function stopOverlayPolling() {
  if (_overlayTimer) {
    clearInterval(_overlayTimer);
    _overlayTimer = null;
  }
  // also clear overlay canvas content so stale boxes don't remain
  try {
    const canvas = document.getElementById('overlay-lane4');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width || 1, canvas.height || 1);
    }
  } catch (e) {}
}
