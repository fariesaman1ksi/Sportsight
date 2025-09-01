# 🏃‍♂️ SportSight – AI Asisten Olahraga untuk Tunanetra

**SportSight** adalah teknologi berbasis **kecerdasan buatan (AI)** yang dirancang khusus untuk membantu penyandang disabilitas **tunanetra** dalam melakukan aktivitas **olahraga** secara **mandiri**, **aman**, dan **nyaman**.

SportSight berfungsi sebagai **“pengganti mata”** dengan memanfaatkan **Computer Vision, GPS, Text-to-Speech, Speech-to-Text, dan Generative AI** untuk memberikan pengalaman olahraga yang lebih **inklusif** dan **bebas hambatan**.

---

## ✨ Fitur Utama

### 👁 Identifikasi Objek & Rintangan  
Menggunakan **YOLOv5** untuk mengenali orang, kendaraan, dan rintangan di sekitar pengguna.

### 🗺 Navigasi Berbasis GPS  
Memberikan instruksi arah secara **real-time** dan **akurat**.

### 🔊 Instruksi Suara Interaktif  
Memanfaatkan **AI Text-to-Speech** untuk menyampaikan informasi visual menjadi audio.

### 🎙 Perintah Suara Cerdas  
Menggunakan **Speech-to-Text (Google Speech Recognition)** untuk memahami perintah pengguna.

### 🧠 Integrasi Generative AI  
Menggunakan **gpt-oss-20b** sebagai **otak SportSight** untuk menafsirkan perintah kompleks.

### 🏃‍♀️ Dukungan Berbagai Aktivitas Atletik  
Mulai dari **berjalan cepat**, **berlari**, hingga **lompat tinggi**, SportSight membantu menjaga arah dan keselamatan.

### ⚡ Respon Cepat & Real-time  
Menggabungkan **computer vision** dan **pemrosesan suara** sehingga sistem merespons secara **instan**.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python 3.9+
- **Library Utama:**
  - `torch` – YOLOv5 inference
  - `opencv-python` – computer vision
  - `speech_recognition` – Speech-to-Text
  - `gTTS` – Text-to-Speech
  - `geopy` – GPS & reverse geocoding
  - `requests` – komunikasi API
- **Model AI:** YOLOv5n + gpt-oss-20b  
- **Platform:** Raspberry Pi + USB Camera

---

## ⚙️ Perangkat Keras yang Dibutuhkan

- Raspberry Pi 4 (4GB RAM)
- Kamera USB / Pi Camera Modul 3
- GPS Module Neo 6M (UART)
- Sensor Ultrasonik HC-SR04
- Speaker / Earphone
- USB Microphone


---

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/muhammadrafyy26-eng/Sportsight.git
cd Sportsight

