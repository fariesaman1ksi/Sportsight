# app.py — SportSight Integrated Dashboard (Dark UI)
# -------------------------------------------------
# Integrates: YOLOv5n (vision), HC-SR04 (ultrasonic), MPU6050 (IMU),
# NEO-6M GPS, STT (Google SpeechRecognition),
# Text generation via OSS 20B (custom REST endpoint), and TTS (gTTS).
#
# This version enhances the CAMERA section: real-time YOLOv5n detection
# with bounding boxes, labels, confidence, and automatic TTS warnings
# when an obstacle is detected within a safety threshold.
#
# Notes:
# - Runs on laptop (simulate hardware) or Raspberry Pi (real HW).
# - Toggle SIMULATE_HARDWARE in sidebar.
# - Provide your OSS20B_API_URL in environment or sidebar.
# -------------------------------------------------

import os
import time
import math
import threading
import queue
from dataclasses import dataclass

import streamlit as st

# --- Optional imports (guarded) ---
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

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
import folium
from streamlit.components.v1 import html
import os

# ----------------------
# Streamlit Page Config
# ----------------------
# Styling Dark Mode untuk Statistik dan Charts
def local_css():
    st.markdown(
        """
        <style>
        /* Gaya umum background dan teks */
        body, .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        /* Metric box dibuat lebih gelap */
        div[data-testid="metric-container"] {
            background-color: #1E222A;
            border: 1px solid #31363F;
            padding: 16px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
        }

        /* Teks metric */
        div[data-testid="metric-container"] label {
            color: #FAFAFA !important;
        }

        /* Nilai metric */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #00E676 !important;
            font-weight: bold;
            font-size: 22px;
        }

        /* Panah naik turun metric */
        svg[data-testid="stMetricDeltaArrow"] {
            stroke: #00E676 !important;
        }

        /* Chart background transparan gelap */
        .css-1aumxhk {
            background-color: #1E222A;
            border-radius: 10px;
            padding: 10px;
        }

        /* Tabel dark mode */
        .dataframe {
            background-color: #1E222A;
            color: #FAFAFA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

st.set_page_config(page_title="SportSight Dashboard", page_icon="🏃", layout="wide")
# CSS Kustom untuk statistik jarak dan yaw
st.markdown("""
    <style>
    /* Gaya umum untuk semua metric box */
    div[data-testid="metric-container"] {
        background-color: #1e1e2f;  /* Warna gelap elegan */
        padding: 15px 10px;
        border-radius: 15px;
        box-shadow: 0px 0px 12px rgba(0, 255, 255, 0.2);
        margin-bottom: 15px;
    }

    /* Gaya untuk label metric (judulnya) */
    div[data-testid="stMetricLabel"] {
        color: #00eaff !important;  /* Label terang */
        font-weight: bold;
    }

    /* Gaya untuk nilai utama metric */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;  /* Nilai putih */
        font-size: 24px;
        font-weight: bold;
    }

    /* Gaya untuk panah indikator */
    svg[data-testid="stMetricDeltaIcon-Up"],
    svg[data-testid="stMetricDeltaIcon-Down"] {
        stroke: #00ffcc !important;  /* Panah neon */
        fill: #00ffcc !important;
    }

    /* Warna teks delta metric */
    div[data-testid="stMetricDelta"] {
        color: #00ffcc !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Dark UI CSS
# ----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
    .stApp {background: radial-gradient(1000px 600px at 20% 10%, #1f1f1f 0%, #0e0e0e 60%, #0a0a0a 100%); color: #eaeaea; font-family: 'Poppins', sans-serif;}
    .glass {backdrop-filter: blur(10px); background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.35);}    
    .brand {font-size: 38px; font-weight: 800; letter-spacing: 1px;}
    .brand-grad {background: linear-gradient(90deg,#ff4b2b,#ff416c,#9b5cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .muted {color:#bdbdbd}
    .tag {display:inline-block; padding:4px 10px; border-radius: 999px; background:#191919; border:1px solid #2a2a2a; margin-right:6px; font-size:12px}
    .ok {color:#78ffb3}
    .warn {color:#ffd166}
    .bad {color:#ff6678}
    div.stButton>button {width:100%; border-radius:12px; padding:10px 14px; font-weight:700; border:0; background:linear-gradient(135deg,#ff4b2b,#ff416c); color:white}
    div.stButton>button:hover {filter:brightness(1.08)}
    textarea, .stTextInput>div>div>input {background:#101010 !important; color:#ddd !important; border-radius:10px}
    .metric {font-size:28px; font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.markdown("<div class='brand brand-grad'>SportSight</div>", unsafe_allow_html=True)
st.sidebar.write("**Mode**")
SIMULATE = st.sidebar.checkbox("Simulasi hardware", value=True, help="Matikan di Raspberry Pi dengan sensor nyata")

OSS20B_URL = st.sidebar.text_input("OSS20B API URL", os.environ.get("OSS20B_API_URL", ""), help="Endpoint REST untuk model 20B Anda, mis: http://ip:port/generate")
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.2, 0.4, 0.1)

st.sidebar.markdown("**YOLOv5n**")
YOLO_PATH = st.sidebar.text_input("Model path/alias", "my_model.pt", help="Gunakan model custom lokal, misal my_model.pt")
CAM_INDEX = st.sidebar.number_input("Camera index", 0, 5, 0)
CONF_THRES = st.sidebar.slider("Confidence", 0.1, 0.9, 0.35, 0.05)

st.sidebar.markdown("**Tujuan GPS**")
TARGET_LAT = st.sidebar.text_input("Target Lat", "-6.2000")
TARGET_LON = st.sidebar.text_input("Target Lon", "106.8166")

# Safety distance threshold (meters)
SAFETY_DISTANCE_M = st.sidebar.slider("Safety distance (m)", 0.5, 5.0, 2.0, 0.1)

# ----------------------
# Queues & State
# ----------------------
ultra_q, imu_q, gps_q, log_q = queue.Queue(1), queue.Queue(1), queue.Queue(1), queue.Queue(200)

def add_log(text: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {text}"
    try:
        log_q.put_nowait(line)
    except queue.Full:
        _ = log_q.get_nowait(); log_q.put_nowait(line)

# ----------------------
# TTS (gTTS) helper
# ----------------------

def speak(text: str):
    add_log(f"TTS: {text}")
    if gTTS is None:
        # fallback: just log
        add_log("gTTS not installed — cannot speak audio")
        return
    try:
        from tempfile import NamedTemporaryFile
        tmp = NamedTemporaryFile(delete=False, suffix=".mp3")
        gTTS(text=text, lang="id").save(tmp.name)
        st.session_state['last_tts'] = tmp.name
    except Exception as e:
        add_log(f"gTTS error: {e}")

# ----------------------
# STT (Google Web Speech)
# ----------------------

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

# ----------------------
# Text Generation (OSS 20B)
# ----------------------

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
ROUTE_HTML = "route.html"

def generate_route_map(gps_points):
    """
    Membuat peta rute dari list koordinat GPS.
    gps_points: [(lat1, lon1), (lat2, lon2), ...]
    """
    if not gps_points:
        return None

    # Pusatkan peta pada titik terakhir
    m = folium.Map(location=gps_points[-1], zoom_start=18, tiles="CartoDB dark_matter")

    # Marker titik awal
    folium.Marker(
        gps_points[0],
        tooltip="Titik Awal",
        icon=folium.Icon(color="green")
    ).add_to(m)

    # Marker titik sekarang
    folium.Marker(
        gps_points[-1],
        tooltip="Titik Sekarang",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Gambar rute
    folium.PolyLine(
        gps_points,
        color="cyan",
        weight=5,
        opacity=0.9
    ).add_to(m)

    # Simpan file route.html
    m.save(ROUTE_HTML)
    return ROUTE_HTML

# ----------------------
# Simple simulated sensor workers (ultrasonic / imu / gps)
# ----------------------

def hcsr_worker(sim=SIMULATE):
    add_log("HC-SR04 worker start")
    if sim or GPIO is None:
        while True:
            dist_m = 1.2 + 0.8 * math.sin(time.time())  # 0.4–2.0 m
            if ultra_q.full():
                _ = ultra_q.get()
            ultra_q.put(dist_m)
            time.sleep(0.2)
    else:
        TRIG, ECHO = 23, 24
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        while True:
            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)
            while GPIO.input(ECHO) == 0:
                pulse_start = time.time()
            while GPIO.input(ECHO) == 1:
                pulse_end = time.time()
            duration = pulse_end - pulse_start
            dist_m = (duration * 343.0) / 2.0
            if ultra_q.full():
                _ = ultra_q.get()
            ultra_q.put(dist_m)
            time.sleep(0.1)


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
        with SMBus(1) as bus:
            addr = 0x68
            bus.write_byte_data(addr, 0x6B, 0)
            while True:
                def read_word(ra):
                    hi = bus.read_byte_data(addr, ra)
                    lo = bus.read_byte_data(addr, ra+1)
                    val = (hi << 8) + lo
                    if val >= 0x8000: val = -((65535 - val) + 1)
                    return val
                ax = read_word(0x3B)/16384.0
                ay = read_word(0x3D)/16384.0
                az = read_word(0x3F)/16384.0
                roll = math.degrees(math.atan2(ay, az))
                pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
                yaw = 0.0
                if imu_q.full(): _ = imu_q.get()
                imu_q.put((yaw, pitch, roll))
                time.sleep(0.1)


def gps_worker(sim=SIMULATE):
    add_log("GPS worker start")
    if sim or serial is None or pynmea2 is None:
        base_lat, base_lon = -6.200, 106.817
        t0 = time.time()
        while True:
            angle = (time.time()-t0)/30.0
            lat = base_lat + 0.0005*math.cos(angle)
            lon = base_lon + 0.0005*math.sin(angle)
            if gps_q.full(): _ = gps_q.get()
            gps_q.put((lat, lon))
            time.sleep(1.0)
    else:
        ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)
        while True:
            line = ser.readline().decode(errors="ignore")
            if line.startswith("$GPRMC") or line.startswith("$GPGGA"):
                try:
                    msg = pynmea2.parse(line)
                    lat = msg.latitude; lon = msg.longitude
                    if gps_q.full(): _ = gps_q.get()
                    gps_q.put((lat, lon))
                except Exception:
                    pass

# Start workers once
if "_threads" not in st.session_state:
    st.session_state["_threads"] = True
    threading.Thread(target=hcsr_worker, daemon=True).start()
    threading.Thread(target=mpu_worker, daemon=True).start()
    threading.Thread(target=gps_worker, daemon=True).start()

# ----------------------
# Header
# ----------------------
colH1, colH2 = st.columns([3,1])
with colH1:
    st.markdown("<div class='brand brand-grad'>🏃‍♂️ SportSight — AI Running Assistant</div>", unsafe_allow_html=True)
    st.markdown("<span class='muted'>Deteksi rintangan (YOLOv5n + HC-SR04), orientasi (MPU6050), arahan rute (GPS), perintah suara (Google STT), respons AI (OSS20B), dan TTS (gTTS).</span>", unsafe_allow_html=True)
with colH2:
    st.markdown("<div class='glass'>" + ''.join([f"<span class='tag'>v5n</span>", f"<span class='tag'>STT</span>", f"<span class='tag'>TTS</span>", f"<span class='tag'>GPS</span>"]) + "</div>", unsafe_allow_html=True)

st.write("")

# ----------------------
# Controls Row
# ----------------------
col1, col2, col3, col4 = st.columns([1.2,1.2,1,1])
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
    st.session_state.setdefault("run_cam", False)
    cam_toggle = st.checkbox("🎥 Kamera (YOLO)", value=st.session_state["run_cam"])
    st.session_state["run_cam"] = cam_toggle

with col4:
    if st.button("🧭 Arahin ke Target"):
        if gps_q.empty():
            speak("Data GPS belum siap.")
        else:
            lat, lon = gps_q.queue[0]
            try:
                tgt_lat = float(TARGET_LAT); tgt_lon = float(TARGET_LON)
                bearing = None
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

# ----------------------
# Camera + YOLOv5n (real-time small loop per rerun)
# ----------------------
@st.cache_resource(show_spinner=False)
def load_yolov5(model_name: str = "my_model.pt"):  # default model custom
    if torch is None:
        add_log("Torch belum terpasang; YOLO tidak tersedia.")
        return None
    try:
        # load YOLO custom model via torch.hub
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_name if model_name.endswith('.pt') else model_name
        )
        model.conf = CONF_THRES
        add_log(f"YOLO custom model loaded: {model_name}")
        return model
    except Exception as e:
        add_log(f"YOLO load fail: {e}")
        return None



import numpy as np

# draw boxes and return any detection summary

def annotate_and_check(frame, results, model, safety_distance_m=SAFETY_DISTANCE_M):
    warn_objects = []
    try:
        # Ambil nama label langsung dari model, bukan dari results
        names = model.names
        r = results.xyxy[0].cpu().numpy()
        h, w = frame.shape[:2]

        for det in r:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = names[int(cls)] if int(cls) in names else str(int(cls))

            # Gambar kotak deteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            txt = f"{label} {conf:.2f}"
            cv2.putText(frame, txt, (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Hitung tinggi bbox untuk cek jarak
            bbox_h = (y2 - y1) / float(h)
            near = False

            # Kalau sensor ultrasonic ada, pakai datanya
            if not ultra_q.empty():
                try:
                    dist = ultra_q.queue[0]
                    if dist <= safety_distance_m:
                        near = True
                except Exception:
                    pass
            else:
                # Kalau nggak ada ultrasonic, pakai heuristik ukuran bbox
                if bbox_h > 0.20:
                    near = True

            if near:
                warn_objects.append((label, float(conf), bbox_h))

    except Exception as e:
        add_log(f"Annotate error: {e}")
    return frame, warn_objects


# === Checkbox Kamera ===
with col3:
    st.session_state.setdefault("run_cam", False)
    st.session_state["run_cam"] = cam_toggle

# === Camera UI & Info Sensor ===
cam_col, info_col = st.columns([2, 1])
with cam_col:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    cam_image_slot = st.empty()
    st.caption("YOLOv5 Live Feed — Detection Overlay")
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    # Status sensor real-time
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("#### Status Sensor")
    dist = ultra_q.queue[0] if not ultra_q.empty() else float('nan')
    yaw, pitch, roll = imu_q.queue[0] if not imu_q.empty() else (float('nan'),) * 3
    gps = gps_q.queue[0] if not gps_q.empty() else (float('nan'), float('nan'))

    st.metric("Jarak depan (m)", f"{dist:0.2f}" if not math.isnan(dist) else "—")
    st.metric("Yaw (°)", f"{yaw:0.0f}" if not math.isnan(yaw) else "—")
    st.metric("GPS", f"{gps[0]:.5f}, {gps[1]:.5f}" if not math.isnan(gps[0]) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

    # AI percakapan
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("#### AI Percakapan")
    st.write("**Perintah:**", st.session_state.get("last_cmd", "—"))
    st.write("**AI:**", st.session_state.get("last_ai", "—"))
    if "last_tts" in st.session_state:
        st.audio(st.session_state["last_tts"], format='audio/mp3')
    st.markdown("</div>", unsafe_allow_html=True)

# === Kamera + YOLO Detection Loop ===
if st.session_state.get('run_cam'):
    if cv2 is None:
        st.warning("⚠️ OpenCV tidak terpasang — kamera tidak tersedia.")
    else:
        model = load_yolov5(YOLO_PATH)
        cap = cv2.VideoCapture(int(CAM_INDEX))

        if not cap.isOpened():
            st.warning("⚠️ Kamera tidak dapat dibuka. Periksa index kamera.")
        else:
            frames = 0
            max_frames = 8

            while frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                warn = []

                if model is not None:
                    try:
                        results = model(img)
                        img, warn = annotate_and_check(img, results, SAFETY_DISTANCE_M)
                    except Exception as e:
                        add_log(f"YOLO inference error: {e}")

                # === Update Detected Labels ===
                detected_labels = [label for label, conf, h in warn]
                if "detected_labels" not in st.session_state:
                    st.session_state["detected_labels"] = []
                st.session_state["detected_labels"].extend(detected_labels)
                st.session_state["detected_labels"] = st.session_state["detected_labels"][-200:]

                # === TTS Warning ===
                if warn:
                    now = time.time()
                    if now - st.session_state.get('last_warn_time', 0) > WARN_COOLDOWN:
                        st.session_state['last_warn_time'] = now
                        top = warn[0]
                        label, conf, _ = top
                        msg = f"Perhatian! {label} terdeteksi di depan. Mohon berhati-hati."
                        threading.Thread(target=speak, args=(msg,), daemon=True).start()
                        add_log(f"Warning triggered: {label} (conf={conf:.2f})")

                # === Tampilkan Frame ===
                cam_image_slot.image(img, channels='RGB', use_column_width=True)
                frames += 1

            cap.release()

# Cooldown for warnings (seconds)
WARN_COOLDOWN = 4.0
if 'last_warn_time' not in st.session_state:
    st.session_state['last_warn_time'] = 0.0

if st.session_state.get('run_cam'):
    if cv2 is None:
        st.warning("OpenCV tidak terpasang — kamera tidak tersedia.")
    else:
        model = load_yolov5(YOLO_PATH)
        cap = cv2.VideoCapture(int(CAM_INDEX))
        if not cap.isOpened():
            st.warning("Kamera tidak dapat dibuka. Periksa index kamera.")
        else:
            # process a small burst of frames per rerun to keep UI responsive
            frames = 0
            max_frames = 8
            while frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                warn = []
                if model is not None:
                    try:
                        results = model(img)
                        img, warn = annotate_and_check(img, results, SAFETY_DISTANCE_M)
                    except Exception as e:
                        add_log(f"YOLO inference error: {e}")
                cam_image_slot.image(img, channels='RGB', use_column_width=True)

                # if there are warnings, voice them (with cooldown)
                if warn:
                    now = time.time()
                    if now - st.session_state['last_warn_time'] > WARN_COOLDOWN:
                        st.session_state['last_warn_time'] = now
                        # build message
                        top = warn[0]
                        label, conf, _ = top
                        msg = f"Perhatian! {label} terdeteksi di depan. Mohon berhati-hati."
                        # run speak in background thread to avoid blocking
                        threading.Thread(target=speak, args=(msg,), daemon=True).start()
                        add_log(f"Warning triggered: {label} (conf={conf:.2f})")
                frames += 1
            cap.release()


# -------------------------------
# Dashboard Statistik & MongoDB Sync
# -------------------------------
import pandas as pd
from datetime import datetime
# ----------------------
# Peta Navigasi GPS
# ----------------------
st.markdown("---")
st.subheader("🗺️ Peta Rute Realtime")

# Ambil data GPS dari queue
if not gps_q.empty():
    gps_data = list(gps_q.queue)
    gps_points = [(lat, lon) for lat, lon in gps_data if lat and lon]
else:
    gps_points = []

# Jika ada data GPS → buat peta
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

st.subheader("📊 Statistik Sensor & AI (Realtime)")

# --- Helper: Ambil snapshot dari queues/ state ---
def snapshot_state():
    """Ambil nilai terakhir sensor/ai dari queue/state"""
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
    # AI state
    last_cmd = st.session_state.get("last_cmd", None)
    last_ai = st.session_state.get("last_ai", None)
    timestamp = datetime.utcnow()
    return {"ts": timestamp, "distance_m": dist, "yaw": yaw, "gps_lat": gps[0], "gps_lon": gps[1], "cmd": last_cmd, "ai": last_ai}

# local history (ke session agar chart terasa realtime)
if "history" not in st.session_state:
    st.session_state["history"] = []

# append current snapshot
snap = snapshot_state()
st.session_state["history"].append(snap)
# keep only recent N
MAX_HISTORY = 200
if len(st.session_state["history"]) > MAX_HISTORY:
    st.session_state["history"] = st.session_state["history"][-MAX_HISTORY:]

# DataFrame untuk charts
df = pd.DataFrame(st.session_state["history"]).set_index("ts")

# Layout: Metrics
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

# ---- Grafik Time Series ----
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

# ---- Metrics di bawah grafik ----
def safe_latest_and_delta(df, col):
    if col not in df.columns or df[col].dropna().empty:
        return None, None
    s = df[col].dropna()
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else last
    return last, (last - prev)

metric_col1, metric_col2 = st.columns(2)

# Metric Jarak
dist_last, dist_delta = safe_latest_and_delta(df, "distance_m")
with metric_col1:
    if dist_last is None:
        st.metric(label="📏 Jarak Sekarang (m)", value="—", delta="—")
    else:
        st.metric(
            label="📏 Jarak Sekarang (m)",
            value=f"{dist_last:.2f}",
            delta=(f"{dist_delta:+.2f} m" if dist_delta is not None else "—")
        )

# Metric Yaw
yaw_last, yaw_delta = safe_latest_and_delta(df, "yaw")
with metric_col2:
    if yaw_last is None:
        st.metric(label="🧭 Yaw Sekarang (°)", value="—", delta="—")
    else:
        st.metric(
            label="🧭 Yaw Sekarang (°)",
            value=f"{yaw_last:.2f}",
            delta=(f"{yaw_delta:+.2f}°" if yaw_delta is not None else "—")
        )

# Object detection summary (count recent labels)
st.markdown("---")
st.markdown("**Ringkasan Deteksi Objek (terakhir)**")
# Hitung label dari sesi deteksi (kita simpan nama di st.session_state['detected_labels'] saat annotate)
labels = st.session_state.get("detected_labels", [])
if labels:
    label_counts = pd.Series(labels).value_counts()
    st.bar_chart(label_counts)
    st.table(label_counts.rename_axis("label").reset_index(name="count"))
else:
    st.info("Belum ada deteksi objek yang disimpan.")

# Tabel log AI/Perintah
st.markdown("---")
st.markdown("**AI / Perintah (Terakhir)**")
last_cmd = st.session_state.get("last_cmd", "—")
last_ai = st.session_state.get("last_ai", "—")
st.write("Perintah:", last_cmd)
st.write("AI:", last_ai)

# show small raw logs from queue
st.markdown("---")
st.markdown("#### Log Aktivitas (ringkas)")
logs = list(log_q.queue)
st.text_area("Log (recent)", value="\n".join(logs[-200:]), height=180)

# ------------------------------
# MongoDB Cloud Sync
# ------------------------------
st.markdown("---")
st.subheader("☁️ Sinkronisasi ke MongoDB Atlas (Cloud)")

st.info("Untuk menyimpan/ambil data sensor & AI di cloud MongoDB, isi connection string (MongoDB URI) di bawah. Gunakan format: mongodb+srv://<user>:<pass>@cluster0.xyz.mongodb.net/test?retryWrites=true&w=majority")

# connection string (env fallback)
MONGO_URI = st.text_input("MongoDB URI (mongodb+srv://...)", os.environ.get("MONGO_URI", ""))
MONGO_DB = st.text_input("Database name", os.environ.get("MONGO_DB", "sportsight_db"))
MONGO_COLL = st.text_input("Collection name", os.environ.get("MONGO_COLL", "telemetry"))

# helper minimal pymongo (lazy import)
def get_mongo_client(uri):
    try:
        from pymongo import MongoClient
    except Exception as e:
        st.error("pymongo belum terpasang. Jalankan: pip install pymongo dnspython")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # quick ping
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
                # insert current snapshot
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
                        # tampilkan time series yang diambil
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

