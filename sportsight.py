#!/usr/bin/env python3
# combined.py — Turn-by-turn, reroute, arrival

import requests
import folium
import serial
import pynmea2
from gtts import gTTS
import os
import uuid
from openai import OpenAI
from geopy.distance import geodesic
import math
import time
import speech_recognition as sr

import subprocess
import cv2
import numpy as np
import torch
import threading
import queue
import RPi.GPIO as GPIO
import re

# ========== CONFIG ==========
api_key = "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"
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)

API_KEY = "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"
HF_BASE = "https://router.huggingface.co/v1"
CHAT_MODEL = "openai/gpt-oss-20b:fireworks-ai"
MIN_CONFIDENCE = 0.3
SPEAK_INTERVAL = 5.0  # YOLO announce interval

client_yolo = OpenAI(base_url=HF_BASE, api_key=API_KEY)

# ====== GPS via GPIO UART ======
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
GPS_POWER_PIN = 17   # optional EN pin
GPS_PPS_PIN   = 18   # optional PPS 1 Hz

# ====== Ultrasonic pins (BCM) ======
TRIG_PIN = 23
ECHO_PIN = 24

# ====== Shared ======
stop_event = threading.Event()

# Navigation state (updated by voice)
nav_update_event = threading.Event()
dest_lock = threading.Lock()
current_destination_name = None
current_destination_coords = None  # (lat, lon)

# Readiness flags
yolo_ready = threading.Event()
gps_ready = threading.Event()
voice_ready = threading.Event()
ultrasonic_ready = threading.Event()

# ========== AUDIO (priority + preempt) ==========
audio_queue = queue.PriorityQueue()
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
                proc = subprocess.Popen(["mpg123", filepath])
                while proc.poll() is None:
                    if pr > 0 and preempt_event.is_set():
                        try: proc.terminate()
                        except Exception: pass
                        break
                    time.sleep(0.05)
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except: pass
                if pr == 0:
                    yolo_active.clear()
                    preempt_event.clear()
            audio_queue.task_done()
        except Exception as e:
            print(f"Audio worker error: {e}")

threading.Thread(target=audio_player_worker, daemon=True).start()

# ========== GPIO INIT ==========
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

# ========== GPS & Geocoding ==========
def get_gps_coordinates(timeout_total=30, debug=False):
    """
    Baca koordinat dari GPS_PORT (GPIO UART). Dukung $GPGGA/$GPRMC/$GNGGA/$GNRMC.
    Kembalikan (lat, lon) atau None jika timeout.
    """
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

    while not stop_event.is_set():
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

# ========== OSRM (walking, fallback driving) ==========
def osrm_route(orig, dest, profile="walking"):
    base = "http://router.project-osrm.org/route/v1"
    url = f"{base}/{profile}/{orig[1]},{orig[0]};{dest[1]},{dest[0]}?overview=full&geometries=geojson&steps=true"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "routes" in data and data["routes"]:
            return data["routes"][0]
    except Exception as e:
        print(f"[OSRM {profile} ERROR] {e}")
    return None

def get_route_structured(orig, dest):
    """
    Return (poly_latlon, maneuvers)
    poly_latlon: list[(lat,lon)]
    maneuvers: list[{'action','modifier','text','loc','distance'}]
    """
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

# ========== Geometry helpers ==========
def meters(p1, p2):
    return geodesic(p1, p2).meters

def bearing_deg(p1, p2):
    # rhumb-ish simple initial bearing
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    brng = (math.degrees(math.atan2(x, y)) + 360) % 360
    return brng

def _proj_xy(lat, lon, lat0):
    # equirectangular projection relative to lat0
    x = (lon) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat) * 110540.0
    return x, y

def point_segment_distance_m(P, A, B):
    # approximate in meters using local equirectangular
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

# ========== Instruction formatting ==========
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
        # default
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
    # “Dalam 200 meter, belok kiri.”
    tts_and_enqueue(f"Dalam {int(dist_m)} meter, {base.lower()}.", priority=1)

def speak_now(base):
    tts_and_enqueue(f"Sekarang {base.lower()}.", priority=1)

def speak_continue(dist_m):
    if dist_m > 20:
        tts_and_enqueue(f"Lanjut lurus {int(dist_m)} meter.", priority=1)

# ====== Map save ======
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

# ====== NAV: destination & main loop ======
def update_destination(dest_name):
    """Ubah tujuan via nama tempat; trigger nav worker."""
    global current_destination_name, current_destination_coords
    coords = get_coordinates_from_name(dest_name)
    if not coords:
        tts_and_enqueue(f"Maaf, tujuan {dest_name} tidak ditemukan.", priority=0)
        return False
    with dest_lock:
        current_destination_name = dest_name
        current_destination_coords = coords
        nav_update_event.set()
    tts_and_enqueue(f"Siap. Navigasi ke {dest_name} dimulai.", priority=0)
    return True

def navigate_like_gmaps(poly, mans, cancel_event=None):
    """
    Turn-by-turn with early/near/now prompts, continue prompts, reroute if off-route,
    until arrive or cancel_event set.
    """
    if not mans:
        tts_and_enqueue("Rute tidak tersedia.", priority=1); return

    # thresholds (walking)
    FAR = 180.0     # meter, notifikasi jauh
    NEAR = 60.0     # meter, notifikasi dekat
    NOW = 15.0      # meter, saat belok
    OFFROUTE = 35.0 # meter, anggap keluar rute
    CONTINUE_GAP = 120.0  # meter, interval “lanjut lurus …”
    RECLAC_MIN_SECS = 10.0

    idx = 0
    spoken_far = set()
    spoken_near = set()
    spoken_now = set()
    last_continue_say = 0.0
    last_recalc = 0.0

    # save initial map
    try:
        # poly already lat,lon
        pass
    except: pass

    # arrival target:
    final_loc = mans[-1]["loc"]

    while not stop_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return

        cur = get_gps_coordinates(timeout_total=5, debug=False)
        if cur is None:
            # no fix yet
            time.sleep(0.5); continue

        # arrival if near final
        if meters(cur, final_loc) <= NOW:
            tts_and_enqueue("Tiba di tujuan.", priority=1)
            return

        # choose current target maneuver (skip arrived/depart that are behind)
        if idx >= len(mans): idx = len(mans)-1
        target = mans[idx]
        dcur = meters(cur, target["loc"])

        # ---- reroute detection ----
        dpoly = distance_to_polyline_m(cur, poly)
        if dpoly > OFFROUTE and (time.time() - last_recalc) > RECLAC_MIN_SECS:
            tts_and_enqueue("Rute diubah, menghitung ulang.", priority=1)
            last_recalc = time.time()
            return "REROUTE"  # signal to worker to recompute

        base = instr_text(target["action"], target["modifier"])

        # ---- staged prompts ----
        if dcur > FAR and idx not in spoken_far:
            speak_ahead(min(300, round(dcur/10)*10), base)  # cap 300 m for walking
            spoken_far.add(idx)

        if NEAR < dcur <= FAR and idx not in spoken_near:
            speak_ahead(int(max(NEAR, round(dcur/10)*10)), base)
            spoken_near.add(idx)

        if dcur <= NOW and idx not in spoken_now:
            speak_now(base)
            spoken_now.add(idx)

        # ---- step advancement ----
        # Advance when closer to next maneuver than current (or already passed)
        if idx + 1 < len(mans):
            dnext = meters(cur, mans[idx+1]["loc"])
            if (dnext + 5) < dcur or (dcur <= NOW/2):
                idx += 1
                # after-turn confirmation: tell how long to next
                if idx < len(mans):
                    seg_len = int(max(20, mans[idx]["distance"]))
                    if seg_len > 40:
                        nowt = time.time()
                        if nowt - last_continue_say > 5:
                            speak_continue(min(seg_len, 400))
                            last_continue_say = nowt
                continue
        else:
            # last step: keep approaching arrival (handled above)
            pass

        # ---- periodic reassurance 'continue' on long stretches ----
        if idx < len(mans):
            seg_len = mans[idx]["distance"]
            nowt = time.time()
            if seg_len >= CONTINUE_GAP and (nowt - last_continue_say) > 40:
                speak_continue(min(int(seg_len), 500))
                last_continue_say = nowt

        time.sleep(0.5)

def gps_nav_worker():
    """Listen for destination changes, compute route, and run nav loop. Reroute on demand."""
    print("📡 GPS Navigation worker aktif. Ucapkan 'menuju ...' untuk set tujuan.")
    while not stop_event.is_set():
        nav_update_event.wait(timeout=0.5)
        if stop_event.is_set(): break
        if not nav_update_event.is_set(): continue

        with dest_lock:
            dest_name = current_destination_name
            dest_coords = current_destination_coords
            nav_update_event.clear()

        # main nav cycle with reroute handling
        while not stop_event.is_set():
            cur = get_gps_coordinates(timeout_total=30, debug=False)
            if cur is None:
                tts_and_enqueue("GPS belum siap, mencari sinyal satelit.", priority=1)
                time.sleep(2); continue

            # compute route
            poly, mans = get_route_structured(cur, dest_coords)
            if not mans:
                tts_and_enqueue("Gagal menghitung rute.", priority=1)
                break

            # save map
            try:
                save_map(poly, cur, dest_coords)
            except: pass

            # run nav loop
            res = navigate_like_gmaps(poly, mans, cancel_event=nav_update_event)
            if res == "REROUTE":
                # loop again to recalc
                continue
            else:
                break  # arrived or cancelled

# ========== YOLO + Ultrasonic + AI ==========
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

def yolo_detection():
    try:
        print("[YOLO] Memuat model YOLOv5n...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        model.eval(); yolo_ready.set()
    except Exception as e:
        print(f"[YOLO LOAD ERROR] {e}"); return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[YOLO STREAM ERROR] Kamera USB tidak bisa dibuka.")
        return
    yolo_ready.set()

    last_announced = 0
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: time.sleep(0.01); continue

            input_frame = cv2.resize(frame, (640, 480))
            with torch.no_grad():
                results = model(input_frame)

            try:
                rendered = results.render()[0]
                cv2.imshow("YOLO Detection", rendered)
            except Exception:
                cv2.imshow("YOLO Detection", input_frame)

            preds = results.pred[0]
            if preds is None or len(preds) == 0:
                if cv2.waitKey(1) & 0xFF == ord('q'): stop_event.set(); break
                continue

            counts = {}
            positions = {}
            frame_w = input_frame.shape[1]

            for p in preds:
                conf = float(p[4]); cls_idx = int(p[5])
                if conf < MIN_CONFIDENCE: continue
                name = results.names.get(cls_idx, str(cls_idx))
                x1, y1, x2, y2 = p[:4]
                x_center = (x1 + x2) / 2
                if x_center < frame_w/3: pos = "kiri"
                elif x_center > (2*frame_w/3): pos = "kanan"
                else: pos = "tengah"
                counts[name] = counts.get(name, 0) + 1
                positions[name] = pos

            if counts:
                now = time.time()
                if now - last_announced >= SPEAK_INTERVAL:
                    parts = []
                    for k, v in counts.items():
                        pos = positions.get(k, "")
                        obj = f"{v} {k}" if v>1 else k
                        if pos: obj += f" di {pos}"
                        parts.append(obj)
                    summary = ", ".join(parts)

                    instr = ""
                    if "kanan" in positions.values(): instr = "Hindari ke kiri."
                    elif "kiri" in positions.values(): instr = "Hindari ke kanan."
                    elif "tengah" in positions.values(): instr = "Hati-hati di depan."

                    jarak = get_distance()
                    if jarak is not None and jarak < 100:
                        prompt = f"Ada {summary}, jaraknya sekitar {jarak} centimeter. {instr} "\
                                 f"Sebutkan informasi itu dengan gaya singkat alami."
                    else:
                        if cv2.waitKey(1) & 0xFF == ord('q'): stop_event.set(); break
                        continue

                    bot_reply = ask_ai(prompt)
                    clean = bot_reply.replace("*","").replace("\n"," ").replace("_"," ").strip()
                    print(f"[AI] {clean}")
                    tts_and_enqueue(clean, priority=0)
                    last_announced = now

            if cv2.waitKey(1) & 0xFF == ord('q'): stop_event.set(); break
    finally:
        cap.release(); cv2.destroyAllWindows()

# ========== VOICE INTENTS ==========
def handle_voice_command(text):
    """
    Intent 1: lokasi → 'lokasi saya di mana', 'di mana saya sekarang', 'saya ada di mana'
    Intent 2: tujuan → 'menuju ...', 'ke ...', 'arah ke ...', 'tujuan ...'
    """
    t = text.lower().strip()

    if re.search(r"(lokasi saya|di mana saya|saya ada di mana)", t):
        loc = get_gps_coordinates(timeout_total=30, debug=False)
        if loc is None:
            tts_and_enqueue("GPS belum siap, mencari sinyal satelit.", priority=0)
            return True
        place = reverse_geocode(loc[0], loc[1])
        tts_and_enqueue(f"Anda berada di {place}.", priority=0)
        return True

    m = re.search(r"^(menuju|ke|arah ke|tujuan)\s+(.+)$", t)
    if m:
        dest_name = m.group(2).strip()
        if dest_name:
            update_destination(dest_name)
            return True
    return False

def voice_listener_worker():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("[VOICE] Mendengarkan mic... bicara kapan saja.")
        voice_ready.set()
    while not stop_event.is_set():
        try:
            with mic as source:
                print("[VOICE] Silakan bicara...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            try:
                query = r.recognize_google(audio, language="id-ID")
                print(f"[VOICE INPUT] {query}")
                if handle_voice_command(query): continue
                # otherwise short assist
                reply = ask_ai(query)
                clean = reply.replace("*","").replace("_"," ").strip()
                tts_and_enqueue(clean, priority=0)
            except sr.UnknownValueError:
                print("[VOICE] Tidak terdengar jelas...")
            except sr.RequestError as e:
                print(f"[VOICE ERROR] {e}")
        except Exception:
            time.sleep(1)

# ========== READINESS MONITOR ==========
def readiness_monitor():
    announced = set(); all_done = False
    def say_once(flag, name, msg):
        if flag.is_set() and name not in announced:
            tts_and_enqueue(msg, priority=0); announced.add(name)
    while not stop_event.is_set():
        say_once(yolo_ready, "yolo", "Visi komputer siap.")
        say_once(gps_ready, "gps", "GPS siap.")
        say_once(voice_ready, "voice", "Pengenalan suara siap.")
        say_once(ultrasonic_ready, "ultra", "Sensor jarak siap.")
        if (not all_done and yolo_ready.is_set() and gps_ready.is_set()
            and voice_ready.is_set() and ultrasonic_ready.is_set()):
            tts_and_enqueue("Semua sistem siap. Ucapkan 'lokasi saya di mana' atau 'menuju ...' untuk mulai.", priority=0)
            all_done = True
        time.sleep(0.2)

# ========== MAIN ==========
if __name__ == "__main__":
    try:
        gpio_setup_common()
        gpio_gps_init()

        t_nav   = threading.Thread(target=gps_nav_worker, daemon=True)
        t_yolo  = threading.Thread(target=yolo_detection, daemon=True)
        t_voice = threading.Thread(target=voice_listener_worker, daemon=True)
        t_ready = threading.Thread(target=readiness_monitor, daemon=True)

        t_nav.start(); t_yolo.start(); t_voice.start(); t_ready.start()

        # OPTIONAL: tujuan awal
        # update_destination("Alun-alun Kota Sukabumi")

        while not stop_event.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[System] Dihentikan oleh user (KeyboardInterrupt).")
        stop_event.set()
    finally:
        try: GPIO.cleanup()
        except: pass
        try: audio_queue.join()
        except: pass
        print("[System] Keluar.")