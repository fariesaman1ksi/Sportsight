import os
import time
import math
import threading
import queue
from datetime import datetime
import pandas as pd
import streamlit as st
import folium
from streamlit.components.v1 import html


# Styling for Streamlit UI
# def local_css():
#     st.markdown(
#         """
#         <style>
#         body, .stApp {
#             background-color: #FFFFFF;
#             color: #FAFAFA;
#             font-family: 'Poppins', sans-serif;
#         }
#         div[data-testid="metric-container"] {
#             background-color: #1E222A;
#             border-radius: 15px;
#             padding: 16px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.4);
#             margin-bottom: 15px;
#         }
#         div[data-testid="metric-container"] label {
#             color: #00eaff !important;
#             font-weight: bold;
#         }
#         div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
#             color: #000000 !important;
#             font-size: 24px;
#             font-weight: bold;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# local_css()

st.set_page_config(page_title="SportSight Dashboard", page_icon="🏃", layout="wide")

# --- Init session_state variables ---
if "history" not in st.session_state:
    st.session_state["history"] = []
if "dummy_merdeka_loaded" not in st.session_state:
    st.session_state["dummy_merdeka_loaded"] = False


# Sidebar Controls
st.sidebar.markdown("<div class='brand brand-grad'>SportSight</div>", unsafe_allow_html=True)
st.sidebar.write("**Mode**")
SIMULATE = st.sidebar.checkbox("Simulasi hardware", value=True, help="Matikan di Raspberry Pi dengan sensor nyata")

OSS20B_URL = st.sidebar.text_input("OSS20B API URL", os.environ.get("OSS20B_API_URL", ""), help="Endpoint REST untuk model 20B Anda")
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.2, 0.4, 0.1)

# GPS Target
TARGET_LAT = st.sidebar.text_input("Target Lat", "-6.2000")
TARGET_LON = st.sidebar.text_input("Target Lon", "106.8166")

# Safety distance threshold (meters)
SAFETY_DISTANCE_M = st.sidebar.slider("Safety distance (m)", 0.5, 5.0, 2.0, 0.1)

# Queues & State
log_q = queue.Queue(200)

if SIMULATE:
    ultra_q, imu_q, gps_q = queue.Queue(1), queue.Queue(1), queue.Queue(1)
    def start_sim():  # dummy fungsi agar tidak error
        pass
    

def add_log(text: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {text}"
    try:
        log_q.put_nowait(line)
    except queue.Full:
        _ = log_q.get_nowait()
        log_q.put_nowait(line)

# TTS (gTTS) helper
def speak(text: str):
    add_log(f"TTS: {text}")
    if gTTS is None:
        add_log("gTTS not installed — cannot speak audio")
        return
    try:
        from tempfile import NamedTemporaryFile
        tmp = NamedTemporaryFile(delete=False, suffix=".mp3")
        gTTS(text=text, lang="id").save(tmp.name)
        st.session_state['last_tts'] = tmp.name
    except Exception as e:
        add_log(f"gTTS error: {e}")

# STT (Google Web Speech)
def listen_google(max_seconds=5, language="id-ID") -> str:
    if sr is None:
        add_log("SpeechRecognition belum terpasang")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        add_log("Mendengarkan perintah...")
        audio = r.listen(source, phrase_time_limit=max_seconds)
    try:
        text = r.recognize_google(audio, language=language)
        add_log(f"STT: {text}")
        return text
    except Exception as e:
        add_log(f"STT gagal: {e}")
        return ""

# Text Generation (OSS 20B)
def generate_reply(prompt: str) -> str:
    if not OSS20B_URL or requests is None:
        # Fallback simple rules
        if "mulai" in prompt.lower():
            return "Menyalakan sistem SportSight dan memulai deteksi."
        if "berhenti" in prompt.lower():
            return "Menghentikan sistem SportSight."
        if "kanan" in prompt.lower():
            return "Belok kanan perlahan."
        if "kiri" in prompt.lower():
            return "Belok kiri perlahan."
        return "Perintah diterima."
    try:
        payload = {"prompt": prompt, "temperature": TEMPERATURE, "max_tokens": 128}
        r = requests.post(OSS20B_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "text" in data:
                return str(data["text"]).strip()
            if "choices" in data and data["choices"]:
                return str(data["choices"][0].get("text", "")).strip()
        return "(tidak ada respons dari OSS20B)"
    except Exception as e:
        add_log(f"OSS20B error: {e}")
        return "Gagal menghubungi model."

# GPS Route Map Generator
ROUTE_HTML = "route.html"
def generate_route_map(gps_points):
    if not gps_points:
        return None
    m = folium.Map(location=gps_points[-1], zoom_start=18, tiles="CartoDB dark_matter")
    folium.Marker(gps_points[0], tooltip="Titik Awal", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps_points[-1], tooltip="Titik Sekarang", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine(gps_points, color="cyan", weight=5, opacity=0.9).add_to(m)
    m.save(ROUTE_HTML)
    return ROUTE_HTML

# Simulated sensor workers
def hcsr_worker(sim=SIMULATE):
    add_log("HC-SR04 worker start")
    if sim or GPIO is None:
        while True:
            dist_m = 1.2 + 0.8 * math.sin(time.time())
            if ultra_q.full():
                _ = ultra_q.get()
            ultra_q.put(dist_m)
            time.sleep(0.2)
    else:
        # Real sensor code omitted for brevity
        pass

def mpu_worker(sim=SIMULATE):
    add_log("MPU6050 worker start")
    if sim or SMBus is None:
        while True:
            yaw = (time.time()*15) % 360
            pitch = 5*math.sin(time.time())
            roll = 4*math.cos(time.time())
            if imu_q.full():
                _ = imu_q.get()
            imu_q.put((yaw, pitch, roll))
            time.sleep(0.2)
    else:
        # Real sensor code omitted for brevity
        pass

def gps_worker(sim=SIMULATE):
    add_log("GPS worker start")
    if sim or serial is None or pynmea2 is None:
        base_lat, base_lon = -6.200, 106.817
        t0 = time.time()
        while True:
            angle = (time.time()-t0)/30.0
            lat = base_lat + 0.0005*math.cos(angle)
            lon = base_lon + 0.0005*math.sin(angle)
            if gps_q.full():
                _ = gps_q.get()
            gps_q.put((lat, lon))
            time.sleep(1.0)
    else:
        # Real sensor code omitted for brevity
        pass

# Start sensor threads once
if "_threads" not in st.session_state:
    st.session_state["_threads"] = True
    if SIMULATE:
        start_sim()  # Menjalankan worker dari dummy.py
    else:
        threading.Thread(target=hcsr_worker, daemon=True).start()
        threading.Thread(target=mpu_worker, daemon=True).start()
        threading.Thread(target=gps_worker, daemon=True).start()


# Header
colH1, colH2 = st.columns([3,1])
with colH1:
    st.title("🏃‍♂️ SportSight — AI Running Assistant")
    st.markdown("<span class='muted'>Deteksi rintangan, orientasi, arahan rute, perintah suara, respons AI, dan TTS.</span>", unsafe_allow_html=True)
st.write("")

# GPS Route Map 
st.markdown("---")
st.subheader("🗺️ Peta Rute Realtime")
#Dummy History
if not st.session_state["dummy_merdeka_loaded"]:
    base_lat, base_lon = -6.9209, 106.9270  # lapdek
    base_time = datetime.utcnow()

    dummy_points = [
        (base_lat, base_lon),                         
        (base_lat+0.0003, base_lon+0.0002),           
        (base_lat+0.0004, base_lon-0.0001),           
        (base_lat+0.0002, base_lon-0.0003),           
        (base_lat, base_lon-0.0004),                  
        (base_lat-0.0003, base_lon-0.0002),           
        (base_lat-0.0004, base_lon+0.0001),           
        (base_lat-0.0002, base_lon+0.0003),           
        (base_lat, base_lon),                         
    ]

    for i, (lat, lon) in enumerate(dummy_points):
        dist = 1.2 if i in [3, 5] else 3.5  
        yaw = (i * 35) % 360 if i in [3, 5] else (i * 20) % 360

        st.session_state["history"].append({
            "ts": base_time.replace(microsecond=0) + pd.Timedelta(seconds=i*20),
            "distance_m": dist,
            "yaw": yaw,
            "gps_lat": lat,
            "gps_lon": lon,
            "cmd": None,
            "ai": None,
        })

    st.session_state["dummy_merdeka_loaded"] = True
    st.success("✅ Dummy history lari di Lapangan Merdeka (GPS, jarak, yaw) sudah dimasukkan.")
    
if st.session_state.get("history"):
    gps_points = [
        (row["gps_lat"], row["gps_lon"]) 
        for row in st.session_state["history"] 
        if row["gps_lat"] is not None and row["gps_lon"] is not None
    ]
    if gps_points:
        route_file = generate_route_map(gps_points)
        if route_file:
            with open(route_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500)
else:
    st.info("Belum ada data GPS untuk ditampilkan.")


# Dashboard Statistik
st.subheader("📊 Statistik Sensor & AI (Realtime)")

def snapshot_state():
    dist = None
    yaw = None
    gps = (None, None)
    if not ultra_q.empty():
        try: dist = float(ultra_q.queue[0])
        except: dist = None
    if not imu_q.empty():
        try: yaw, pitch, roll = imu_q.queue[0]
        except: yaw = None
    if not gps_q.empty():
        try: gps = gps_q.queue[0]
        except: gps = (None, None)
    last_cmd = st.session_state.get("last_cmd", None)
    last_ai = st.session_state.get("last_ai", None)
    timestamp = datetime.utcnow()
    return {"ts": timestamp, "distance_m": dist, "yaw": yaw, "gps_lat": gps[0], "gps_lon": gps[1], "cmd": last_cmd, "ai": last_ai}

if "history" not in st.session_state:
    st.session_state["history"] = []

snap = snapshot_state()
if snap["distance_m"] is not None or snap["yaw"] is not None or snap["gps_lat"] is not None:
    st.session_state["history"].append(snap)
MAX_HISTORY = 200
if len(st.session_state["history"]) > MAX_HISTORY:
    st.session_state["history"] = st.session_state["history"][-MAX_HISTORY:]

df = pd.DataFrame(st.session_state["history"]).set_index("ts")

m1, m2, m3 = st.columns(3)
with m1:
    latest_dist = df["distance_m"].dropna().iloc[-1] if "distance_m" in df.columns and not df["distance_m"].dropna().empty else None
    st.metric("Jarak depan (m)", f"{latest_dist:.2f}" if latest_dist is not None else "—")
with m2:
    latest_yaw = df["yaw"].dropna().iloc[-1] if "yaw" in df.columns and not df["yaw"].dropna().empty else None
    st.metric("Yaw (°)", f"{latest_yaw:.0f}" if latest_yaw is not None else "—")
with m3:
    latest_gps = (df["gps_lat"].dropna().iloc[-1], df["gps_lon"].dropna().iloc[-1]) if ("gps_lat" in df.columns and not df["gps_lat"].dropna().empty) else (None, None)
    st.metric("GPS (lat,lon)", f"{latest_gps[0]:.5f}, {latest_gps[1]:.5f}" if latest_gps[0] is not None else "—")

st.markdown("---")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**📊 Jarak (m) — Time Series**")
    if "distance_m" in df.columns and not df["distance_m"].dropna().empty:
        st.line_chart(df["distance_m"].ffill(), use_container_width=True)
    else:
        st.info("Belum ada data jarak untuk ditampilkan.")

with chart_col2:
    st.markdown("**🧭 Yaw (°) — Time Series**")
    if "yaw" in df.columns and not df["yaw"].dropna().empty:
        st.line_chart(df["yaw"].ffill(), use_container_width=True)
    else:
        st.info("Belum ada data yaw untuk ditampilkan.")

def safe_latest_and_delta(df, col):
    if col not in df.columns or df[col].dropna().empty:
        return None, None
    s = df[col].dropna()
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else last
    return last, (last - prev)

metric_col1, metric_col2 = st.columns(2)
dist_last, dist_delta = safe_latest_and_delta(df, "distance_m")
with metric_col1:
    if dist_last is None:
        st.metric(label="📏 Jarak Sekarang (m)", value="—", delta="—")
    else:
        st.metric(label="📏 Jarak Sekarang (m)", value=f"{dist_last:.2f}", delta=(f"{dist_delta:+.2f} m" if dist_delta is not None else "—"))

yaw_last, yaw_delta = safe_latest_and_delta(df, "yaw")
with metric_col2:
    if yaw_last is None:
        st.metric(label="🧭 Yaw Sekarang (°)", value="—", delta="—")
    else:
        st.metric(label="🧭 Yaw Sekarang (°)", value=f"{yaw_last:.2f}", delta=(f"{yaw_delta:+.2f}°" if yaw_delta is not None else "—"))




