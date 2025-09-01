# 🧭 Smart Blind Navigation Assistant

Proyek ini adalah **asisten navigasi pintar** untuk membantu **penyandang tunanetra** menggunakan **Raspberry Pi**.  
Sistem ini menggabungkan **GPS, Computer Vision (YOLOv5), Voice Command, OpenAI/HuggingFace AI, dan sensor ultrasonik** untuk memberikan arahan secara **audio** dalam bahasa Indonesia.

---

## ✨ Fitur Utama
- 📍 **Navigasi GPS Real-time**  
  Memberikan instruksi belok dan pemberitahuan saat mendekati tujuan.
- 🎙 **Kontrol Suara**  
  Pengguna bisa memberikan perintah seperti:
  - “Lokasi saya di mana?”
  - “Menuju Alun-alun Kota Sukabumi”
- 🧠 **Deteksi Objek dengan YOLOv5**  
  Mendeteksi pejalan kaki, kendaraan, dan rintangan di sekitar pengguna.
- 🔊 **Pemberitahuan Audio Interaktif**  
  Menggunakan **gTTS** untuk mengumumkan instruksi dan peringatan.
- 📡 **Pengenalan Lingkungan**  
  Memberi tahu lokasi pengguna menggunakan **reverse geocoding**.
- 🚧 **Sensor Ultrasonik**  
  Mengukur jarak rintangan terdekat untuk menghindari tabrakan.

---

## 🛠️ Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python 3.9+
- **Library Utama:**
  - [`requests`](https://docs.python-requests.org/)
  - [`folium`](https://python-visualization.github.io/folium/)
  - [`pynmea2`](https://github.com/Knio/pynmea2) – parsing data GPS
  - [`geopy`](https://geopy.readthedocs.io/)
  - [`opencv-python`](https://opencv.org/) – untuk streaming kamera
  - [`torch`](https://pytorch.org/) – YOLOv5 inference
  - [`speech_recognition`](https://pypi.org/project/SpeechRecognition/)
  - [`gTTS`](https://pypi.org/project/gTTS/) – text-to-speech
- **Model AI:** YOLOv5n + OpenAI/HuggingFace API
- **Platform:** Raspberry Pi + USB Camera

---

## ⚙️ Perangkat Keras yang Dibutuhkan
- Raspberry Pi 4 (4GB RAM)
- Pi Camera Modul 3
- GPS Module Neo 6M (UART)
- Sensor Ultrasonik HC-SR04
- Speaker atau Earphone
- USB Microphone

---

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/muhammadrafyy26-eng/Sportsight.git
cd Sportsight

