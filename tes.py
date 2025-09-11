import streamlit as st
import pandas as pd
import psutil
from datetime import datetime
from pymongo import MongoClient
from bson.binary import Binary
import io
from PIL import Image

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="SportSight Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# MongoDB Connection
# ==============================
@st.cache_resource
def init_connection():
    # Ganti sesuai MongoDB Atlas atau lokal
    client = MongoClient("mongodb://localhost:27017/")
    return client

client = init_connection()
db = client["sportsight"]
collection = db["images"]

# ==============================
# Ambil Gambar Terbaru
# ==============================
def get_latest_image():
    doc = collection.find_one(sort=[("_id", -1)])  # ambil dokumen terbaru
    if doc and "image" in doc:
        img_bytes = doc["image"]  # disimpan sebagai Binary
        return Image.open(io.BytesIO(img_bytes))
    return None

# ==============================
# Dummy Data Generator (sementara)
# ==============================
def get_dummy_detection():
    return pd.DataFrame({
        "Object": ["person", "bicycle", "car"],
        "Confidence": [0.92, 0.85, 0.76],
        "Estimated Distance (m)": [5.2, 12.3, 20.5]
    })

def get_dummy_gps():
    return {
        "Latitude": -7.2575,
        "Longitude": 112.7521,
        "Speed (km/h)": 4.5,
        "Heading": "North-East"
    }

def get_dummy_sensor():
    return {
        "Accelerometer": [0.01, -0.02, 9.81],
        "Gyroscope": [0.001, 0.002, 0.003],
        "Ultrasonic Distance (m)": 2.4,
        "Battery (%)": 85
    }

def get_dummy_audio():
    return {
        "User Command (STT)": "Mulai navigasi",
        "System Response (TTS)": "5 meter lagi ada sepeda di depan"
    }

# ==============================
# Dashboard Layout
# ==============================
st.title("🏃 SportSight Dashboard")
st.caption("Monitoring real-time data dari perangkat SportSight (gambar dari MongoDB)")

# 1️⃣ Gambar dari MongoDB + Deteksi Objek
st.markdown("## 🎥 Hasil Deteksi Objek")
col1, col2 = st.columns([2,1])

with col1:
    img = get_latest_image()
    if img:
        st.image(img, caption="📷 Gambar terbaru dari SportSight", use_container_width=True)
    else:
        st.warning("⚠️ Belum ada gambar di MongoDB")

with col2:
    df = get_dummy_detection()
    st.dataframe(df.style.background_gradient(cmap="Greens"), use_container_width=True)

# 2️⃣ GPS Data
st.markdown("## 🗺️ Lokasi & Navigasi")
gps = get_dummy_gps()
st.json(gps)

# 3️⃣ Sensor Data
st.markdown("## 📡 Data Sensor")
st.json(get_dummy_sensor())

# 4️⃣ Audio
st.markdown("## 🎙️ Audio Interaction")
audio = get_dummy_audio()
st.info(f"🗣️ {audio['User Command (STT)']}")
st.success(f"🔊 {audio['System Response (TTS)']}")

# 5️⃣ Performance
st.markdown("## ⚡ Performance Sistem")
cpu = psutil.cpu_percent()
ram = psutil.virtual_memory().percent
col1, col2, col3 = st.columns(3)
col1.metric("FPS", 28)
col2.metric("CPU Usage", f"{cpu}%")
col3.metric("RAM Usage", f"{ram}%")

# Footer
st.markdown("---")
st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")