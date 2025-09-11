import math
import time
import threading
import queue

# Antrian data untuk disimulasikan
ultra_q = queue.Queue(1)
imu_q = queue.Queue(1)
gps_q = queue.Queue(1)

# Worker simulasi jarak HC-SR04
def hcsr_sim_worker():
    while True:
        dist_m = 1.0 + 0.5 * math.sin(time.time())
        if ultra_q.full():
            _ = ultra_q.get()
        ultra_q.put(dist_m)
        time.sleep(0.2)

# Worker simulasi orientasi MPU6050
def mpu_sim_worker():
    while True:
        yaw = (time.time() * 15) % 360
        pitch = 5 * math.sin(time.time())
        roll = 4 * math.cos(time.time())
        if imu_q.full():
            _ = imu_q.get()
        imu_q.put((yaw, pitch, roll))
        time.sleep(0.2)

# Worker simulasi GPS
def gps_sim_worker():
    base_lat, base_lon = -6.200, 106.817
    t0 = time.time()
    while True:
        angle = (time.time() - t0) / 30.0
        lat = base_lat + 0.0005 * math.cos(angle)
        lon = base_lon + 0.0005 * math.sin(angle)
        if gps_q.full():
            _ = gps_q.get()
        gps_q.put((lat, lon))
        time.sleep(1.0)

# Fungsi untuk mulai semua worker simulasi
def start_simulation():
    threading.Thread(target=hcsr_sim_worker, daemon=True).start()
    threading.Thread(target=mpu_sim_worker, daemon=True).start()
    threading.Thread(target=gps_sim_worker, daemon=True).start()

# Bisa jalan sendiri kalau dijalankan langsung
if __name__ == "__main__":
    print("Menjalankan simulasi sensor...")
    start_simulation()
    while True:
        time.sleep(5)
        print("Distance:", ultra_q.queue[0] if not ultra_q.empty() else "—")
        print("IMU:", imu_q.queue[0] if not imu_q.empty() else "—")
        print("GPS:", gps_q.queue[0] if not gps_q.empty() else "—")
