#!/usr/bin/env python3
# newsportsight.py — STT+LLM+TTS (selalu on), operasi berat (YOLO+Ultrasonic+GPS Nav) start/stop via suara

import os
import uuid
import time
import math
import re
import queue
import threading
import subprocess
from collections import deque

import requests

import serial
import pynmea2
from gtts import gTTS
from openai import OpenAI
from geopy.distance import geodesic

import cv2
import numpy as np
import torch
import speech_recognition as sr
import RPi.GPIO as GPIO

# ===== Kompas & IMU =====
import smbus2  # I2C untuk HMC5883L & MPU6050

# --- Alamat I2C & register (HMC5883L + MPU6050) ---
I2C_BUS = 1
# HMC5883L
HMC_ADDR = 0x1E
REG_CONFIG_A = 0x00
REG_CONFIG_B = 0x01
REG_MODE     = 0x02
REG_X_MSB    = 0x03
MODE_CONTINUOUS = 0x00
# MPU6050
MPU_ADDR = 0x68
MPU_PWR_MGMT_1 = 0x6B
MPU_ACCEL_XOUT_H = 0x3B

# Kalibrasi magnetometer (silakan sesuaikan)
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0
SCALE_X  = 1.0
SCALE_Y  = 1.0
SCALE_Z  = 1.0

# Parameter heading
DECLINATION_DEG   = 0.0
HEADING_ALPHA     = 0.25
HEADING_BIAS_DEG  = 0.0
AUTO_SET_BIAS_ON_START = True

# Ambang arah benar/keluar
ALIGN_OK_DEG      = 25.0
ALIGN_EXIT_DEG    = 35.0
ALIGN_STABLE_SEC  = 1.0

INVERT_TURN       = False

class MPU6050:
    def _init_(self, bus_id=1, addr=MPU_ADDR):
        self.bus = smbus2.SMBus(bus_id)
        self.addr = addr
        self.bus.write_byte_data(self.addr, MPU_PWR_MGMT_1, 0x00)
        time.sleep(0.05)

    def _read_i16(self, reg_hi):
        hi = self.bus.read_byte_data(self.addr, reg_hi)
        lo = self.bus.read_byte_data(self.addr, reg_hi + 1)
        val = (hi << 8) | lo
        if val > 32767: val -= 65536
        return val

    def read_accel_g(self):
        ax = self._read_i16(MPU_ACCEL_XOUT_H) / 16384.0
        ay = self._read_i16(MPU_ACCEL_XOUT_H + 2) / 16384.0
        az = self._read_i16(MPU_ACCEL_XOUT_H + 4) / 16384.0
        return ax, ay, az

    def read_pitch_roll_rad(self):
        ax, ay, az = self.read_accel_g()
        roll  = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        return pitch, roll

class TiltCompass:
    def _init_(self, bus_id=1, addr=HMC_ADDR):
        self.bus = smbus2.SMBus(bus_id)
        self.addr = addr
        self._smooth = None
        self.bus.write_byte_data(self.addr, REG_CONFIG_A, 0x70)  # 8avg,15Hz
        self.bus.write_byte_data(self.addr, REG_CONFIG_B, 0xA0)  # Gain
        self.bus.write_byte_data(self.addr, REG_MODE, MODE_CONTINUOUS)
        time.sleep(0.06)
        self.imu = MPU6050(bus_id=bus_id)

    def _read_raw_axis(self, start_addr):
        hi = self.bus.read_byte_data(self.addr, start_addr)
        lo = self.bus.read_byte_data(self.addr, start_addr + 1)
        val = (hi << 8) | lo
        if val > 32767: val -= 65536
        return val

    def read_mag_xyz(self):
        x = self._read_raw_axis(REG_X_MSB)
        z = self._read_raw_axis(REG_X_MSB + 2)
        y = self._read_raw_axis(REG_X_MSB + 4)
        x = (x - OFFSET_X) * SCALE_X
        y = (y - OFFSET_Y) * SCALE_Y
        z = (z - OFFSET_Z) * SCALE_Z
        return x, y, z

    def read_heading_deg(self):
        pitch, roll = self.imu.read_pitch_roll_rad()
        mx, my, mz = self.read_mag_xyz()
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll),  math.sin(roll)
        Xh = mx*cp + mz*sp
        Yh = mx*sr*sp + my*cr - mz*sr*cp
        hdg = math.degrees(math.atan2(Yh, Xh))
        if hdg < 0: hdg += 360.0
        hdg = (hdg + DECLINATION_DEG) % 360.0
        if self._smooth is None:
            self._smooth = hdg
        else:
            diff = (hdg - self._smooth + 540) % 360 - 180
            self._smooth = (self._smooth + HEADING_ALPHA * diff) % 360.0
        return (self._smooth + HEADING_BIAS_DEG) % 360.0

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def rel_bearing_text(rel_brg):
    if rel_brg <= 15 or rel_brg >= 345: return "Lurus"
    if 15 < rel_brg <= 45:              return "Sedikit ke kanan"
    if 45 < rel_brg <= 135:             return "Belok kanan"
    if 135 < rel_brg <= 225:            return "Putar balik"
    if 225 < rel_brg <= 315:            return "Belok kiri"
    if 315 < rel_brg < 345:             return "Sedikit ke kiri"
    return "Lurus"

tilt_compass = None

# ========================== CONFIG COMMON ==========================
API_KEY = "REMOVED"
HF_BASE = "https://router.huggingface.co/v1"
CHAT_MODEL = "openai/gpt-oss-20b:fireworks-ai"
client_yolo = OpenAI(base_url=HF_BASE, api_key=API_KEY)

SPEAK_INTERVAL = 5.0
OSRM_TIMEOUT = 6
REROUTE_GPS_TIMEOUT = 6

# ====== GPS via GPIO UART ======
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
GPS_POWER_PIN = 17
GPS_PPS_PIN   = 18

# ====== Ultrasonic pins (BCM) ======
TRIG_PIN = 23
ECHO_PIN = 24

# ====== FLAGS / LOCKS ======
stop_event = threading.Event()
ops_stop_event = threading.Event()
ops_running = threading.Event()

nav_update_event = threading.Event()
dest_lock = threading.Lock()
current_destination_name = None
current_destination_coords = None  # (lat, lon)

# Readiness flags
yolo_ready = threading.Event()
gps_ready = threading.Event()
voice_ready = threading.Event()
ultrasonic_ready = threading.Event()

# ========================== AUDIO (priority + preempt) ==========================
import queue as _queue
audio_queue = _queue.PriorityQueue()
preempt_event = threading.Event()
yolo_active = threading.Event()

_current_proc_lock = threading.Lock()
_current_proc = None

def _set_current_proc(p):
    global _current_proc
    with _current_proc_lock:
        _current_proc = p

def _get_current_proc():
    with _current_proc_lock:
        return _current_proc

def tts_and_enqueue(text, priority=1):
    try:
        filename = f"/tmp/ai_tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang="id")
        tts.save(filename)
        if priority == 0:
            yolo_active.set()
            preempt_event.set()
        audio_queue.put((priority, filename))
    except Exception as e:
        print(f"TTS enqueue error: {e}")

def audio_player_worker():
    while True:
        try:
            pr, filepath = audio_queue.get()
            if filepath is None:
                break
            if os.path.exists(filepath):
                proc = subprocess.Popen(["mpg123", "-q", filepath])
                _set_current_proc(proc)
                while proc.poll() is None:
                    if pr > 0 and preempt_event.is_set():
                        try: proc.terminate()
                        except Exception: pass
                        break
                    time.sleep(0.05)
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                if pr == 0:
                    yolo_active.clear()
                    preempt_event.clear()
            audio_queue.task_done()
        except Exception as e:
            print(f"Audio worker error: {e}")

threading.Thread(target=audio_player_worker, daemon=True).start()

# ========================== GPIO INIT ==========================
def gpio_setup_common():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    try:
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
    except Exception as e:
        print(f"[GPIO SETUP ERROR] {e}")

def gpio_gps_init():
    try:
        GPIO.setup(GPS_POWER_PIN, GPIO.OUT, initial=GPIO.HIGH)
        print("[GPIO GPS POWER] ON (GPIO17 HIGH)")
    except Exception as e:
        print(f"[GPIO GPS POWER] skip: {e}")
    try:
        GPIO.setup(GPS_PPS_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(GPS_PPS_PIN, GPIO.RISING, bouncetime=5)
        print("[GPIO PPS] PPS detect aktif di GPIO18")
    except Exception as e:
        print(f"[GPIO PPS] skip: {e}")

def wait_pps(timeout=5):
    t0 = time.time()
    try:
        while time.time()-t0 < timeout and not stop_event.is_set():
            if GPIO.event_detected(GPS_PPS_PIN):
                print("[PPS] Edge terdeteksi.")
                return True
            time.sleep(0.05)
    except Exception:
        pass
    return False

# ========================== GPS & Geocoding ==========================
def get_gps_coordinates(timeout_total=30, debug=False):
    try:
        GPIO.output(GPS_POWER_PIN, GPIO.HIGH)
    except Exception:
        pass

    _ = wait_pps(timeout=3)

    try:
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
        if debug: print(f"[GPS] Buka {GPS_PORT} @ {GPS_BAUD}")
    except Exception as e:
        print(f"[GPS ERROR] Tidak bisa buka {GPS_PORT}: {e}")
        return None

    wanted = ("$GPGGA", "$GPRMC", "$GNGGA", "$GNRMC")
    t0 = time.time()

    while not stop_event.is_set() and not ops_stop_event.is_set():
        if time.time() - t0 > timeout_total:
            if debug: print("[GPS] Timeout total tanpa fix.")
            return None
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            if debug: print(f"[GPS READ ERROR] {e}")
            time.sleep(0.2); continue

        if not line: continue
        if debug and line.startswith(("$", "!")):
            print("[GPS RAW]", line[:120])

        if line.startswith(wanted):
            try:
                msg = pynmea2.parse(line)
                lat, lon = getattr(msg, "latitude", None), getattr(msg, "longitude", None)
                gps_qual = getattr(msg, "gps_qual", None)
                status   = getattr(msg, "status", None)
                if lat and lon and float(lat)!=0 and float(lon)!=0:
                    ok = ((gps_qual is None and status is None) or
                          (gps_qual and int(gps_qual)>0) or
                          (status and status.upper()=="A"))
                    if ok:
                        gps_ready.set()
                        return float(lat), float(lon)
            except Exception as e:
                if debug: print(f"[GPS PARSE ERROR] {e}")
                continue
        time.sleep(0.05)

def gps_warmup_worker():
    print("🛰️  GPS warm-up worker aktif.")
    while not stop_event.is_set():
        if gps_ready.is_set():
            time.sleep(5); continue
        loc = get_gps_coordinates(timeout_total=15, debug=False)
        if loc is not None:
            if not gps_ready.is_set():
                tts_and_enqueue("GPS siap.", priority=0)
            gps_ready.set()
            time.sleep(10)
        else:
            time.sleep(3)

def get_coordinates_from_name(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "RaspberryPi-Navigator/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"[GEOCODE ERROR] {e}")
    return None

def reverse_geocode(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "jsonv2"}
    headers = {"User-Agent": "RaspberryPi-Navigator/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        j = r.json()
        name = j.get("name") or ""
        display = j.get("display_name") or name
        return display or f"{lat:.5f}, {lon:.5f}"
    except Exception as e:
        print(f"[REVERSE GEOCODE ERROR] {e}")
        return f"{lat:.5f}, {lon:.5f}"

# ========================== OSRM ==========================
def osrm_route(orig, dest, profile="walking", timeout=OSRM_TIMEOUT):
    base = "http://router.project-osrm.org/route/v1"
    url = f"{base}/{profile}/{orig[1]},{orig[0]};{dest[1]},{dest[0]}?overview=full&geometries=geojson&steps=true"
    try:
        r = requests.get(url, timeout=timeout)
        data = r.json()
        if isinstance(data, dict) and data.get("routes"):
            return data["routes"][0]
    except Exception as e:
        print(f"[OSRM {profile} ERROR] {e}")
    return None

def get_route_structured(orig, dest):
    data = osrm_route(orig, dest, profile="walking") or osrm_route(orig, dest, profile="driving")
    if not data:
        return [], []
    coords = data["geometry"]["coordinates"]  # [lon,lat]
    poly_latlon = [(lat, lon) for lon, lat in coords]
    mans = []
    for leg in data.get("legs", []):
        for step in leg.get("steps", []):
            man = step.get("maneuver", {})
            action = man.get("type", "") or ""
            modifier = man.get("modifier", "") or ""
            loc = (man["location"][1], man["location"][0])  # lat, lon
            dist = float(step.get("distance", 0.0))
            text = step.get("name","")
            mans.append({
                "action": action, "modifier": modifier, "text": text,
                "loc": loc, "distance": dist
            })
    return poly_latlon, mans

# ========================== Geometry helpers ==========================
def meters(p1, p2):
    return geodesic(p1, p2).meters

def _proj_xy(lat, lon, lat0):
    x = (lon) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat) * 110540.0
    return x, y

def point_segment_distance_m(P, A, B):
    lat0 = (A[0]+B[0]+P[0])/3.0
    Px, Py = _proj_xy(P[0], P[1], lat0)
    Ax, Ay = _proj_xy(A[0], A[1], lat0)
    Bx, By = _proj_xy(B[0], B[1], lat0)
    ABx, ABy = (Bx-Ax, By-Ay)
    APx, APy = (Px-Ax, Py-Ay)
    ab2 = ABx*ABx + ABy*ABy
    if ab2 == 0:
        dx, dy = APx, APy
    else:
        t = max(0.0, min(1.0, (APx*ABx + APy*ABy)/ab2))
        projx, projy = Ax + t*ABx, Ay + t*ABy
        dx, dy = Px - projx, Py - projy
    return math.hypot(dx, dy)

def distance_to_polyline_m(P, poly):
    if len(poly) < 2: return float("inf")
    mind = float("inf")
    for i in range(len(poly)-1):
        d = point_segment_distance_m(P, poly[i], poly[i+1])
        if d < mind: mind = d
    return mind

# ========================== Instruction formatting ==========================
def instr_text(action, modifier):
    action = (action or "").lower()
    modifier = (modifier or "").lower()
    if action == "arrive":
        return "Tiba di tujuan."
    if action == "roundabout":
        return "Masuk bundaran, ikuti petunjuk."
    if action in ("turn","continue","fork","end of road","merge"):
        dir_map = {
            "left": "belok kiri",
            "slight left": "belok kiri ringan",
            "sharp left": "belok kiri tajam",
            "right": "belok kanan",
            "slight right": "belok kanan ringan",
            "sharp right": "belok kanan tajam",
            "uturn": "putar balik",
            "straight": "lurus"
        }
        if modifier in dir_map:
            return dir_map[modifier].capitalize()
        if action == "continue":
            return "Lurus"
        if action == "fork":
            return "Ambil percabangan yang sesuai"
        if action == "merge":
            return "Gabung ke jalur"
        return "Ikuti jalan"
    if action == "depart":
        return "Mulai perjalanan"
    return (action or "Ikuti jalan").capitalize()

def speak_ahead(dist_m, base):
    tts_and_enqueue(f"Dalam {int(dist_m)} meter, {base.lower()}.", priority=1)

def speak_now(base):
    tts_and_enqueue(f"Sekarang {base.lower()}.", priority=1)

def speak_continue(dist_m):
    if dist_m > 20:
        tts_and_enqueue(f"Lanjut lurus {int(dist_m)} meter.", priority=1)

# ========================== Map save ==========================
def save_map(route_coords, orig, dest):
    try:
        m = folium.Map(location=orig, zoom_start=16)
        folium.Marker(orig, tooltip="Start (GPS)").add_to(m)
        folium.Marker(dest, tooltip="Destination").add_to(m)
        if route_coords:
            folium.PolyLine(route_coords, color="blue", weight=5).add_to(m)
        m.save("route.html")
    except Exception as e:
        print(f"[MAP ERROR] {e}")

# ========================== NAV: destination & main loop ==========================
def update_destination(dest_name):
    global current_destination_name, current_destination_coords
    coords = get_coordinates_from_name(dest_name)
    if not coords:
        tts_and_enqueue(f"Maaf, tujuan {dest_name} tidak ditemukan.", priority=0)
        return False
    with dest_lock:
        current_destination_name = dest_name
        current_destination_coords = coords
        nav_update_event.set()
    tts_and_enqueue(f"Siap. Navigasi ke {dest_name} disiapkan.", priority=0)
    return True

def navigate_with_turn_triggers(poly, mans, cancel_event=None):
    if not mans:
        tts_and_enqueue("Rute tidak tersedia.", priority=1)
        return

    FAR = 180.0
    NEAR_TRIGGER = 60.0
    NOW_TRIGGER  = 7.0
    CONTINUE_GAP = 120.0

    MIN_TURN_AHEAD_M = 15.0
    OFFROUTE_BASE = 60.0
    OFFROUTE_STREAK_NEED = 6
    START_GRACE_SEC = 20.0
    RECLAC_MIN_SECS = 10.0

    idx = 0
    last_continue_say = 0.0
    last_recalc = 0.0

    start_ts = time.time()
    recent_pts = deque(maxlen=6)
    offroute_streak = 0

    spoken_far = set()
    spoken_now = set()

    final_loc = mans[-1]["loc"]

    align_state_aligned = False
    align_enter_time = None
    last_heading_tts = 0.0
    HEADING_TTS_COOLDOWN = 3.0
    awaiting_post_turn_align = False

    def _avg_point(pts):
        lat = sum(p[0] for p in pts) / len(pts)
        lon = sum(p[1] for p in pts) / len(pts)
        return (lat, lon)

    def _jitter_radius_m(pts):
        if len(pts) < 3:
            return 0.0
        c = _avg_point(pts)
        d = [meters(p, c) for p in pts]
        return sum(d) / len(d)

    while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
        cur = get_gps_coordinates(timeout_total=5, debug=False)
        if cur is None:
            time.sleep(0.5)
            continue

        recent_pts.append(cur)
        cur_smooth = _avg_point(recent_pts) if len(recent_pts) >= 3 else cur

        if meters(cur_smooth, final_loc) <= NOW_TRIGGER:
            tts_and_enqueue("Tiba di tujuan.", priority=1)
            return

        if idx >= len(mans):
            idx = len(mans) - 1

        target = mans[idx]
        base = instr_text(target["action"], target["modifier"])
        dcur = meters(cur_smooth, target["loc"])

        jitter = _jitter_radius_m(recent_pts)
        OFFROUTE = OFFROUTE_BASE + max(15.0, 3.0 * jitter)
        dpoly = distance_to_polyline_m(cur_smooth, poly)

        try:
            if tilt_compass is not None:
                head = tilt_compass.read_heading_deg()
                brg  = bearing_deg(cur_smooth[0], cur_smooth[1], target["loc"][0], target["loc"][1])
                rel_nominal = (brg - head + 360.0) % 360.0
                rel = rel_nominal if not INVERT_TURN else (head - brg + 360.0) % 360.0
                now_ts = time.time()
                base_is_straight = base.lower().startswith("lurus")

                if rel <= ALIGN_OK_DEG or rel >= (360.0 - ALIGN_OK_DEG):
                    if not align_state_aligned:
                        if align_enter_time is None:
                            align_enter_time = now_ts
                        elif (now_ts - align_enter_time) >= ALIGN_STABLE_SEC:
                            align_state_aligned = True
                            align_enter_time = None
                            if awaiting_post_turn_align:
                                tts_and_enqueue("Lurus di jalur.", priority=1)
                                awaiting_post_turn_align = False
                            else:
                                tts_and_enqueue("Arah benar, lanjut lurus.", priority=1)
                            last_heading_tts = now_ts
                else:
                    if align_state_aligned and (rel > ALIGN_EXIT_DEG and rel < (360.0 - ALIGN_EXIT_DEG)):
                        align_state_aligned = False
                        align_enter_time = None
                    if (not base_is_straight) and (now_ts - last_heading_tts >= HEADING_TTS_COOLDOWN):
                        txt = rel_bearing_text(rel)
                        if txt != "Lurus":
                            tts_and_enqueue(txt + ".", priority=1)
                            last_heading_tts = now_ts
        except Exception:
            pass

        bl = base.lower()
        is_turn = (target["action"] in ("turn", "roundabout", "fork", "end of road", "merge")
                   or "belok" in bl or "putar" in bl or "bundaran" in bl)

        if dcur <= max(NEAR_TRIGGER, MIN_TURN_AHEAD_M) and idx not in spoken_far:
            ahead_dist = max(MIN_TURN_AHEAD_M, int(round(dcur)))
            if is_turn:
                speak_ahead(ahead_dist, base)
            else:
                speak_ahead(max(5, int(round(dcur / 5) * 5)), base)
            spoken_far.add(idx)

        if dcur <= NOW_TRIGGER and idx not in spoken_now:
            speak_now(base)
            spoken_now.add(idx)
            if is_turn:
                awaiting_post_turn_align = True

        if (time.time() - start_ts) > START_GRACE_SEC:
            if dpoly > OFFROUTE:
                offroute_streak += 1
            else:
                offroute_streak = 0
            if (offroute_streak >= OFFROUTE_STREAK_NEED and
                (time.time() - last_recalc) > RECLAC_MIN_SECS):
                tts_and_enqueue("Rute diubah, menghitung ulang.", priority=1)
                last_recalc = time.time()
                return "REROUTE"

        if idx + 1 < len(mans):
            dnext = meters(cur_smooth, mans[idx+1]["loc"])
            if (dnext + 5) < dcur or dcur <= NOW_TRIGGER/2:
                idx += 1
                if idx < len(mans):
                    seg_len = int(max(20, mans[idx]["distance"]))
                    if seg_len > 40:
                        speak_continue(min(seg_len, 400))
                        time.sleep(0.2)

        time.sleep(0.5)

def gps_nav_worker(cancel_event=None):
    print("📡 GPS Navigation worker aktif (standby). Ucapkan 'menuju ...' lalu 'mulai' untuk menyalakan navigasi.")
    last_poly = None
    last_mans = None
    auto_bias_done = False

    while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
        nav_update_event.wait(timeout=0.5)
        if stop_event.is_set() or (cancel_event and cancel_event.is_set()):
            break
        if not nav_update_event.is_set():
            continue

        with dest_lock:
            dest_name = current_destination_name
            dest_coords = current_destination_coords
            nav_update_event.clear()

        while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
            cur = get_gps_coordinates(timeout_total=30, debug=False)
            if cur is None:
                tts_and_enqueue("GPS belum siap, mencari sinyal satelit.", priority=1)
                time.sleep(2)
                continue

            poly, mans = get_route_structured(cur, dest_coords)
            if not mans:
                tts_and_enqueue("Gagal menghitung rute.", priority=1)
                break

            last_poly, last_mans = poly, mans
            try: save_map(poly, cur, dest_coords)
            except Exception: pass

            try:
                if AUTO_SET_BIAS_ON_START and (not auto_bias_done) and tilt_compass is not None and mans:
                    head0 = tilt_compass.read_heading_deg()
                    brg0  = bearing_deg(cur[0], cur[1], mans[0]["loc"][0], mans[0]["loc"][1])
                    global HEADING_BIAS_DEG
                    HEADING_BIAS_DEG = (brg0 - head0 + 360.0) % 360.0
                    if HEADING_BIAS_DEG > 180.0: HEADING_BIAS_DEG -= 360.0
                    auto_bias_done = True
                    tts_and_enqueue("Kompas dikalibrasi terhadap arah rute.", priority=1)
            except Exception:
                pass

            res = navigate_with_turn_triggers(poly, mans, cancel_event=cancel_event)
            if res != "REROUTE":
                break

            cur2 = get_gps_coordinates(timeout_total=REROUTE_GPS_TIMEOUT, debug=False) or cur
            new_route = osrm_route(cur2, dest_coords, profile="walking", timeout=OSRM_TIMEOUT) \
                        or osrm_route(cur2, dest_coords, profile="driving", timeout=OSRM_TIMEOUT)
            if new_route and "geometry" in new_route:
                coords = new_route["geometry"]["coordinates"]
                poly = [(lat, lon) for lon, lat in coords]
                mans = []
                for leg in new_route.get("legs", []):
                    for step in leg.get("steps", []):
                        man = step.get("maneuver", {})
                        action = man.get("type", "") or ""
                        modifier = man.get("modifier", "") or ""
                        loc = (man["location"][1], man["location"][0])
                        dist = float(step.get("distance", 0.0))
                        text = step.get("name", "")
                        mans.append({
                            "action": action, "modifier": modifier, "text": text,
                            "loc": loc, "distance": dist
                        })
                last_poly, last_mans = poly, mans
                try: save_map(poly, cur2, dest_coords)
                except Exception: pass
            else:
                tts_and_enqueue("Jaringan lambat, menggunakan rute sebelumnya.", priority=1)
                if last_poly and last_mans:
                    poly, mans = last_poly, last_mans
                else:
                    break
            continue

# ========================== Ultrasonic ==========================
def get_distance():
    try:
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.000002)
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        timeout = time.time() + 0.02
        start_time = None; end_time = None

        while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
            start_time = time.time()
        timeout = time.time() + 0.02
        while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
            end_time = time.time()

        if start_time is None or end_time is None:
            return None

        duration = end_time - start_time
        distance = round((duration * 34300) / 2, 2)  # cm
        if distance > 0: ultrasonic_ready.set()
        return distance
    except Exception as e:
        print(f"[ULTRASONIC ERROR] {e}")
        return None

# ========================== Chat / AI Prompt ==========================
messages_lock = threading.Lock()
messages = [
    {"role": "system",
     "content": ("Kamu adalah asisten navigasi untuk orang buta. "
                 "Selalu bicara singkat: sebutkan objek, jarak, dan cara menghindar. "
                 "Jangan berbasa-basi, jangan menjelaskan panjang. "
                 "Contoh: 'Orang di depan 80 cm, geser kiri.' "
                 "atau 'Mobil di depan 40 cm, berhenti.' "
                 "Dahulukan instruksi dibanding percakapan.")}
]

def ask_ai(user_text):
    with messages_lock:
        messages.append({"role": "user", "content": user_text})
    try:
        completion = client_yolo.chat.completions.create(
            model=CHAT_MODEL, messages=messages
        )
        bot_reply = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_reply})
        return bot_reply
    except Exception as e:
        print(f"[AI ERROR] {e}")
        return "Maaf, terjadi kesalahan."

# ========================== YOLO FAST (Ultralytics) ==========================
from ultralytics import YOLO
from threading import Thread, Lock

MODEL_PATH   = "models/my_model.pt"   # <<< sesuai permintaan kamu
IMG_SIZE     = 320
CONF_THRES   = 0.50
MAX_DET      = 20
FRAME_SKIP   = 2
DEVICE       = "cpu"
CPU_THREADS  = 2

# ===== Kamera (V4L2 /dev/video0 – TANPA libcamera) =====
CAM_SRC      = 0
CAM_WIDTH    = 640
CAM_HEIGHT   = 480
CAM_FPS_HINT = 30
USE_MJPG     = True
SHOW_WINDOW  = True
MIN_CONFIDENCE = CONF_THRES

def set_cpu_threads(n: int):
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass

class CamReader:
    """Threaded camera reader (V4L2 /dev/video0, tanpa libcamera)."""
    def _init_(self, src=0, width=640, height=480, fps=30, use_mjpg=True):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if use_mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError("Kamera tidak bisa dibuka! Pastikan /dev/video0 ada (aktifkan bcm2835-v4l2).")

        self.ok = True
        self.frame = None
        self.lock = Lock()
        self.t = Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.002); continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        return (False, None) if f is None else (True, f)

    def release(self):
        self.ok = False
        try: self.t.join(timeout=0.5)
        except Exception: pass
        self.cap.release()

def draw_boxes(frame, results, names, thickness=2):
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None: continue
        try:
            b = boxes.xyxy.cpu().numpy().astype(int)
            c = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
        except Exception:
            b = np.array(boxes.xyxy, dtype=int)
            c = np.array(boxes.cls, dtype=int)
            conf = np.array(boxes.conf, dtype=float)
        for (x1, y1, x2, y2), ci, cf in zip(b, c, conf):
            if cf < CONF_THRES: continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)
            label_name = names.get(ci, str(ci)) if isinstance(names, dict) else str(ci)
            label = f"{label_name} {cf:.2f}"
            y_text = y1 - 8 if y1 > 18 else y1 + 18
            cv2.putText(frame, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    return frame

def yolo_detection(cancel_event=None):
    try:
        set_cpu_threads(CPU_THREADS)

        print("[YOLO] Memuat model:", MODEL_PATH)
        model = YOLO(MODEL_PATH)

        names = {}
        if hasattr(model, "names") and isinstance(model.names, (list, dict)):
            names = model.names
        elif hasattr(model, "model") and hasattr(model.model, "names"):
            names = model.model.names

        dummy = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        _ = model.predict(dummy, imgsz=IMG_SIZE, conf=CONF_THRES,
                          max_det=MAX_DET, device=DEVICE, verbose=False)

        cam = CamReader(src=CAM_SRC, width=CAM_WIDTH, height=CAM_HEIGHT,
                        fps=CAM_FPS_HINT, use_mjpg=USE_MJPG)
        yolo_ready.set()

        last_results = None
        frame_id = 0
        fps_hist = deque(maxlen=20)
        last_announced = 0.0

        if SHOW_WINDOW:
            try: cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
            except Exception: pass

        while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.005); continue

            if frame_id % max(1, FRAME_SKIP) == 0:
                t0 = time.time()
                results = model.predict(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRES,
                    max_det=MAX_DET, device=DEVICE, verbose=False
                )
                last_results = results
                fps = 1.0 / max(1e-6, (time.time() - t0))
                fps_hist.append(fps)

                counts = {}
                positions = {}
                h, w = frame.shape[:2]
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None: continue
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls  = boxes.cls.cpu().numpy().astype(int)
                        conf = boxes.conf.cpu().numpy()
                    except Exception:
                        xyxy = np.array(boxes.xyxy)
                        cls  = np.array(boxes.cls, dtype=int)
                        conf = np.array(boxes.conf, dtype=float)
                    for (x1, y1, x2, y2), ci, cf in zip(xyxy, cls, conf):
                        if cf < MIN_CONFIDENCE: continue
                        name = names.get(ci, str(ci))
                        xc = (x1 + x2) / 2.0
                        pos = "kiri" if xc < w/3 else ("kanan" if xc > 2*w/3 else "tengah")
                        counts[name] = counts.get(name, 0) + 1
                        positions[name] = pos

                if counts and (time.time() - last_announced) >= SPEAK_INTERVAL:
                    parts = []
                    for k, v in counts.items():
                        pos = positions.get(k, "")
                        obj = f"{v} {k}" if v > 1 else k
                        if pos: obj += f" di {pos}"
                        parts.append(obj)
                    summary = ", ".join(parts)

                    instr = ""
                    if "kanan" in positions.values(): instr = "Hindari ke kiri."
                    elif "kiri" in positions.values(): instr = "Hindari ke kanan."
                    elif "tengah" in positions.values(): instr = "Hati-hati di depan."

                    jarak = get_distance()
                    if jarak is not None and jarak < 200:
                        prompt = (f"Ada {summary}, jaraknya sekitar {int(jarak)} centimeter. "
                                  f"{instr} Sebutkan informasi itu dengan gaya singkat alami.")
                        bot_reply = ask_ai(prompt)
                        clean = bot_reply.replace("*","").replace("\n"," ").replace("_"," ").strip()
                        print(f"[AI] {clean}")
                        tts_and_enqueue(clean, priority=0)
                        last_announced = time.time()

            if last_results is not None:
                frame = draw_boxes(frame, last_results, names, thickness=2)

            if fps_hist:
                fps_disp = sum(fps_hist) / len(fps_hist)
                cv2.putText(frame, f"FPS~{fps_disp:.1f} (skip={FRAME_SKIP})",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if SHOW_WINDOW:
                try:
                    cv2.imshow("YOLO Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if cancel_event: cancel_event.set()
                        break
                except Exception:
                    pass
            else:
                time.sleep(0.001)

            frame_id += 1

    except Exception as e:
        print(f"[YOLO ERROR] {e}")
    finally:
        try: cam.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

# ========================== VOICE INTENTS & CONTROL ==========================
t_nav = None
t_yolo = None

def start_ops():
    """Mulai operasi berat: YOLO + GPS Nav. Dipanggil saat 'mulai'."""
    global t_nav, t_yolo
    if ops_running.is_set():
        tts_and_enqueue("Operasi sudah berjalan.", priority=0)
        return
    ops_stop_event.clear()
    ops_running.set()

    t_yolo = threading.Thread(target=yolo_detection, kwargs={"cancel_event": ops_stop_event}, daemon=True)
    t_yolo.start()

    t_nav  = threading.Thread(target=gps_nav_worker, kwargs={"cancel_event": ops_stop_event}, daemon=True)
    t_nav.start()

    tts_and_enqueue("Operasi dimulai. Deteksi objek dan navigasi aktif.", priority=0)

def stop_ops():
    """Hentikan operasi berat, tetap sisakan STT+LLM+TTS."""
    global t_nav, t_yolo
    if not ops_running.is_set():
        tts_and_enqueue("Operasi sudah berhenti. Mode siaga.", priority=0)
        return
    ops_stop_event.set()
    ops_running.clear()
    try:
        if t_yolo and t_yolo.is_alive(): t_yolo.join(timeout=1.5)
    except Exception: pass
    try:
        if t_nav and t_nav.is_alive(): t_nav.join(timeout=1.5)
    except Exception: pass
    yolo_ready.clear()
    ultrasonic_ready.clear()
    tts_and_enqueue("Operasi dihentikan. Sistem kembali ke mode siaga.", priority=0)

def handle_voice_command(text):
    """
    Intent 0: kontrol → 'mulai', 'stop', 'berhenti'
    Intent 1: lokasi → 'lokasi saya di mana', 'di mana saya sekarang', 'saya ada di mana'
    Intent 2: tujuan → 'menuju ...', 'ke ...', 'arah ke ...', 'tujuan ...'
    """
    t = text.lower().strip()

    if re.search(r"^(mulai|start|aktifkan)$", t):
        start_ops(); return True
    if re.search(r"^(stop|berhenti|nonaktifkan|pause)$", t):
        stop_ops(); return True

    if re.search(r"(lokasi saya|di mana saya|saya ada di mana)", t):
        loc = get_gps_coordinates(timeout_total=15, debug=False)
        if loc is None:
            tts_and_enqueue("GPS belum siap, mencari sinyal satelit.", priority=0)
            return True
        place = reverse_geocode(loc[0], loc[1])
        tts_and_enqueue(f"Anda berada di {place}.", priority=0)
        return True

    m = re.search(r"^(menuju|ke|arah ke|tujuan|pergi ke)\s+(.+)$", t)
    if m:
        dest_name = m.group(2).strip()
        if dest_name:
            update_destination(dest_name)
            if not ops_running.is_set():
                tts_and_enqueue("Tujuan disimpan. Ucapkan 'mulai' untuk memulai navigasi.", priority=0)
            return True
    return False

def voice_listener_worker():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    r.energy_threshold = 2500
    r.pause_threshold = 0.6
    r.non_speaking_duration = 0.2

    mic = sr.Microphone(sample_rate=16000)
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("[VOICE] Mendengarkan mic (mode siaga)...")
        voice_ready.set()
        tts_and_enqueue("Sistem siap. Ucapkan 'mulai' untuk menyalakan deteksi dan navigasi, atau 'stop' untuk berhenti.", priority=0)
    while not stop_event.is_set():
        try:
            with mic as source:
                print("[VOICE] Silakan bicara...")
                audio = r.listen(source, timeout=6, phrase_time_limit=8)
            try:
                query = r.recognize_google(audio, language="id-ID")
                print(f"[VOICE INPUT] {query}")
                if handle_voice_command(query):
                    continue
                reply = ask_ai(query)
                clean = reply.replace("*","").replace("_"," ").replace("\n"," ").strip()
                tts_and_enqueue(clean, priority=0)
            except sr.UnknownValueError:
                print("[VOICE] Tidak terdengar jelas...")
            except sr.RequestError as e:
                print(f"[VOICE ERROR] {e}")
        except Exception:
            time.sleep(1)

# ========================== READINESS MONITOR ==========================
def readiness_monitor():
    announced = set()
    while not stop_event.is_set():
        if gps_ready.is_set() and "gps" not in announced:
            tts_and_enqueue("GPS siap.", priority=0); announced.add("gps")
        if voice_ready.is_set() and "voice" not in announced:
            announced.add("voice")
        if ops_running.is_set():
            if yolo_ready.is_set() and "yolo" not in announced:
                tts_and_enqueue("Visi komputer siap.", priority=0); announced.add("yolo")
            if ultrasonic_ready.is_set() and "ultra" not in announced:
                tts_and_enqueue("Sensor jarak siap.", priority=0); announced.add("ultra")
        time.sleep(0.5)

# ========================== MAIN ==========================
if _name_ == "_main_":
    try:
        gpio_setup_common()
        gpio_gps_init()

        try:
            tilt_compass = TiltCompass(bus_id=I2C_BUS, addr=HMC_ADDR)
            print("[COMPASS] TiltCompass aktif.")
        except Exception as e:
            tilt_compass = None
            print(f"[COMPASS WARN] Kompas tidak tersedia: {e}")

        t_voice = threading.Thread(target=voice_listener_worker, daemon=True)
        t_ready = threading.Thread(target=readiness_monitor, daemon=True)
        t_gpswu = threading.Thread(target=gps_warmup_worker, daemon=True)

        t_voice.start()
        t_ready.start()
        t_gpswu.start()

        # OPTIONAL: pre-set destination
        # update_destination("SMA Negeri 1 Sukabumi")

        while not stop_event.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[System] Dihentikan oleh user (KeyboardInterrupt).")
        stop_event.set()
        ops_stop_event.set()
    finally:
        try: GPIO.cleanup()
        except Exception: pass
        try: audio_queue.join()
        except Exception: pass
        print("[System] Keluar.")