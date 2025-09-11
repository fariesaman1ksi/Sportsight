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

# --- Optional imports (guarded) ---
try:
    import RPi.GPIO as GPIO  # Raspberry Pi only
except Exception:
    GPIO = None

try:
    from smbus2 import SMBus
except Exception:
    SMBus = None

try:
    import serial, pynmea2
except Exception:
    serial = None
    pynmea2 = None

try:
    import speech_recognition as sr  # Google Web Speech API
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import requests
except Exception:
    requests = None

# Styling Dark Mode for Streamlit UI
def local_css():
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Poppins', sans-serif;
        }
        div[data-testid="metric-container"] {
            background-color: #1E222A;
            border-radius: 15px;
            padding: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
            margin-bottom: 15px;
        }
        div[data-testid="metric-container"] label {
            color: #00eaff !important;
            font-weight: bold;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
local_css()

st.set_page_config(page_title="SportSight Dashboard", page_icon="🏃", layout="wide")

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
ultra_q, imu_q, gps_q, log_q = queue.Queue(1), queue.Queue(1), queue.Queue(1), queue.Queue(200)

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
    threading.Thread(target=hcsr_worker, daemon=True).start()
    threading.Thread(target=mpu_worker, daemon=True).start()
    threading.Thread(target=gps_worker, daemon=True).start()

# Header
colH1, colH2 = st.columns([3,1])
with colH1:
    st.markdown("<div class='brand brand-grad'>🏃‍♂️ SportSight — AI Running Assistant</div>", unsafe_allow_html=True)
    st.markdown("<span class='muted'>Deteksi rintangan, orientasi, arahan rute, perintah suara, respons AI, dan TTS.</span>", unsafe_allow_html=True)
with colH2:
    st.markdown("<div class='glass'>" + ''.join([f"<span class='tag'>STT</span>", f"<span class='tag'>TTS</span>", f"<span class='tag'>GPS</span>"]) + "</div>", unsafe_allow_html=True)

st.write("")

# Controls Row
col1, col2, col3 = st.columns([1.5,1.5,1])
with col1:
    if st.button("🎙️ Dengarkan Perintah"):
        cmd = listen_google()
        if cmd:
            reply = generate_reply(cmd)
            st.session_state["last_cmd"] = cmd
            st.session_state["last_ai"] = reply
            speak(reply)

with col2:
    if st.button("🗣️ Ucapkan Status"):
        dist = ultra_q.queue[0] if not ultra_q.empty() else None
        pos = gps_q.queue[0] if not gps_q.empty() else None
        yaw, pitch, roll = imu_q.queue[0] if not imu_q.empty() else (0,0,0)
        msg = []
        if dist is not None:
            msg.append(f"Rintangan {dist:.1f} meter di depan.")
        if pos is not None:
            msg.append(f"Lokasi {pos[0]:.5f}, {pos[1]:.5f}.")
        msg.append(f"Orientasi yaw {yaw:.0f} derajat.")
        speak(" ".join(msg))

with col3:
    if st.button("🧭 Arahin ke Target"):
        if gps_q.empty():
            speak("Data GPS belum siap.")
        else:
            lat, lon = gps_q.queue[0]
            try:
                tgt_lat = float(TARGET_LAT)
                tgt_lon = float(TARGET_LON)
                def calc_bearing(lat1, lon1, lat2, lon2):
                    phi1, phi2 = math.radians(lat1), math.radians(lat2)
                    dlon = math.radians(lon2 - lon1)
                    x = math.sin(dlon) * math.cos(phi2)
                    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlon)
                    brng = (math.degrees(math.atan2(x, y)) + 360) % 360
                    return brng
                bearing = calc_bearing(lat, lon, tgt_lat, tgt_lon)
                speak(f"Arahkan badan ke {bearing:.0f} derajat dan maju.")
            except Exception:
                speak("Koordinat target tidak valid.")

# Sensor Status and AI Conversation UI
info_col = st.columns([1])[0]
with info_col:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("#### Status Sensor")
    dist = ultra_q.queue[0] if not ultra_q.empty() else float('nan')
    yaw, pitch, roll = imu_q.queue[0] if not imu_q.empty() else (float('nan'),) * 3
    gps = gps_q.queue[0] if not gps_q.empty() else (float('nan'), float('nan'))

    st.metric("Jarak depan (m)", f"{dist:0.2f}" if not math.isnan(dist) else "—")
    st.metric("Yaw (°)", f"{yaw:0.0f}" if not math.isnan(yaw) else "—")
    st.metric("GPS", f"{gps[0]:.5f}, {gps[1]:.5f}" if not math.isnan(gps[0]) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("#### AI Percakapan")
    st.write("**Perintah:**", st.session_state.get("last_cmd", "—"))
    st.write("**AI:**", st.session_state.get("last_ai", "—"))
    if "last_tts" in st.session_state:
        st.audio(st.session_state["last_tts"], format='audio/mp3')
    st.markdown("</div>", unsafe_allow_html=True)

# GPS Route Map
st.markdown("---")
st.subheader("🗺️ Peta Rute Realtime")

if not gps_q.empty():
    gps_data = list(gps_q.queue)
    gps_points = [(lat, lon) for lat, lon in gps_data if lat and lon]
else:
    gps_points = []

if gps_points:
    route_html = generate_route_map(gps_points)
    if route_html and os.path.exists(route_html):
        with open(route_html, "r", encoding="utf-8") as f:
            route_html_code = f.read()
        html(route_html_code, height=500)
    else:
        st.error("Gagal membuat peta rute.")
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

# AI / Perintah Terakhir
st.markdown("---")
st.markdown("**AI / Perintah (Terakhir)**")
last_cmd = st.session_state.get("last_cmd", "—")
last_ai = st.session_state.get("last_ai", "—")
st.write("Perintah:", last_cmd)
st.write("AI:", last_ai)

# Log Aktivitas Ringkas
st.markdown("---")
st.markdown("#### Log Aktivitas (ringkas)")
logs = list(log_q.queue)
st.text_area("Log (recent)", value="\n".join(logs[-200:]), height=180)

# MongoDB Cloud Sync
st.markdown("---")
st.subheader("☁️ Sinkronisasi ke MongoDB Atlas (Cloud)")

st.info("Untuk menyimpan/ambil data sensor & AI di cloud MongoDB, isi connection string (MongoDB URI) di bawah. Gunakan format: mongodb+srv://<user>:<pass>@cluster0.xyz.mongodb.net/test?retryWrites=true&w=majority")

MONGO_URI = st.text_input("MongoDB URI (mongodb+srv://...)", os.environ.get("MONGO_URI", ""))
MONGO_DB = st.text_input("Database name", os.environ.get("MONGO_DB", "sportsight_db"))
MONGO_COLL = st.text_input("Collection name", os.environ.get("MONGO_COLL", "telemetry"))

def get_mongo_client(uri):
    try:
        from pymongo import MongoClient
    except Exception:
        st.error("pymongo belum terpasang. Jalankan: pip install pymongo dnspython")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"Gagal koneksi MongoDB: {e}")
        return None

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🔁 Kirim snapshot sekarang (Push)"):
        if not MONGO_URI:
            st.error("Isi MongoDB URI terlebih dahulu.")
        else:
            client = get_mongo_client(MONGO_URI)
            if client:
                db = client[MONGO_DB]
                coll = db[MONGO_COLL]
                doc = snapshot_state()
                doc["_ts"] = datetime.utcnow()
                try:
                    coll.insert_one(doc)
                    st.success("Snapshot berhasil disimpan ke MongoDB.")
                except Exception as e:
                    st.error(f"Gagal insert: {e}")

with col2:
    if st.button("📥 Ambil 100 data terakhir (Pull)"):
        if not MONGO_URI:
            st.error("Isi MongoDB URI terlebih dahulu.")
        else:
            client = get_mongo_client(MONGO_URI)
            if client:
                db = client[MONGO_DB]
                coll = db[MONGO_COLL]
                try:
                    docs = list(coll.find().sort("_ts", -1).limit(100))
                    if docs:
                        pull_df = pd.DataFrame(docs)
                        if "distance_m" in pull_df.columns:
                            st.markdown("Chart jarak dari DB (pulled)")
                            pull_df = pull_df.set_index("_ts")
                            st.line_chart(pull_df["distance_m"].astype(float).ffill())
                        st.dataframe(pull_df.head(50))
                    else:
                        st.info("Tidak ada data di collection.")
                except Exception as e:
                    st.error(f"Gagal query: {e}")

with col3:
    if st.button("⚙️ Setup Index & Test"):
        if not MONGO_URI:
            st.error("Isi MongoDB URI terlebih dahulu.")
        else:
            client = get_mongo_client(MONGO_URI)
            if client:
                db = client[MONGO_DB]
                coll = db[MONGO_COLL]
                try:
                    coll.create_index([("_ts", -1)])
                    coll.create_index([("distance_m", 1)])
                    st.success("Index dibuat/test OK.")
                except Exception as e:
                    st.error(f"Gagal buat index: {e}")

st.markdown("---")
st.caption("Tip: simpan MONGO_URI di environment variable atau file .env agar tidak menuliskannya ke UI di production.")
