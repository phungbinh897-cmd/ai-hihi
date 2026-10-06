"""
╔══════════════════════════════════════════════════════════════════╗
║         PARTICLE SPHERE 3D — KONTROL TANGAN INTERAKTIF          ║
║         OpenCV + MediaPipe + NumPy                              ║
╚══════════════════════════════════════════════════════════════════╝

Instalasi:
    pip install opencv-python mediapipe numpy

Cara pakai:
    python particle_sphere.py

Kontrol:
    • 2 tangan terdeteksi  → Renggangkan/rapatkan untuk zoom
    • 1 tangan terdeteksi  → Bola mengikuti posisi tangan
    • Q / ESC              → Keluar
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI GLOBAL
# ═══════════════════════════════════════════════════════════════════

JUMLAH_PARTIKEL    = 900     # Lebih banyak = lebih detail, lebih berat
RADIUS_MIN         = 70      # Radius minimum bola (px di layar)
RADIUS_MAX         = 300     # Radius maksimum bola (px di layar)
RADIUS_DEFAULT     = 150     # Radius awal

SPEED_ROT_Y        = 0.013   # Kecepatan rotasi sumbu Y (radian/frame)
SPEED_ROT_X        = 0.007   # Kecepatan rotasi sumbu X (radian/frame)

FOCAL_LENGTH       = 700     # Panjang fokus perspektif (px)
Z_OFFSET           = 800     # Offset jarak kamera (z)

LERP_RADIUS        = 0.09    # Kecepatan interpolasi radius (0-1, makin besar makin cepat)
LERP_POS           = 0.10    # Kecepatan interpolasi posisi bola

WARNA_SHIFT_SPEED  = 40      # Kecepatan pergeseran hue (derajat/detik)

# ═══════════════════════════════════════════════════════════════════
#  INISIALISASI MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils

detektor_tangan = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.60
)

# ═══════════════════════════════════════════════════════════════════
#  PEMBANGKIT PARTIKEL BOLA (Fibonacci Lattice)
# ═══════════════════════════════════════════════════════════════════

def buat_partikel_bola(n: int) -> np.ndarray:
    """
    Membuat N titik yang tersebar merata di permukaan bola unit
    menggunakan metode Fibonacci Sphere / Golden Spiral.

    Hasilnya adalah array (N, 3) dengan koordinat ternormalisasi
    sehingga setiap titik berjarak 1.0 dari pusat bola.
    """
    golden = (1 + math.sqrt(5)) / 2  # Rasio emas ≈ 1.618
    titik  = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        # Sudut polar: arccos memberi distribusi merata di sumbu Y
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        # Sudut azimut berbasis golden ratio untuk pola spiral
        phi   = 2 * math.pi * i / golden

        titik[i, 0] = math.sin(theta) * math.cos(phi)  # X
        titik[i, 1] = math.sin(theta) * math.sin(phi)  # Y
        titik[i, 2] = math.cos(theta)                   # Z

    return titik


# ═══════════════════════════════════════════════════════════════════
#  MATRIKS ROTASI 3D
# ═══════════════════════════════════════════════════════════════════

def rot_y(a: float) -> np.ndarray:
    """Matriks rotasi 3×3 terhadap sumbu Y sebesar sudut a (radian)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def rot_x(a: float) -> np.ndarray:
    """Matriks rotasi 3×3 terhadap sumbu X sebesar sudut a (radian)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  PROYEKSI PERSPEKTIF 3D → 2D
# ═══════════════════════════════════════════════════════════════════

def proyeksi(pts3d: np.ndarray, cx: int, cy: int):
    """
    Memproyeksikan titik-titik 3D ke koordinat layar 2D.

    Args:
        pts3d : Array (N,3) — titik dalam ruang 3D
        cx, cy: Pusat layar (pixel)

    Returns:
        px, py : Array koordinat layar (int)
        z_norm : Kedalaman ternormalisasi [0,1] (0=dekat, 1=jauh)
        skala  : Faktor skala perspektif per titik
    """
    z_cam  = pts3d[:, 2] + Z_OFFSET               # Z relatif terhadap kamera
    z_cam  = np.maximum(z_cam, 1.0)               # Hindari pembagian nol

    skala  = FOCAL_LENGTH / z_cam                 # Makin jauh → makin kecil

    px = (pts3d[:, 0] * skala + cx).astype(np.int32)
    py = (pts3d[:, 1] * skala + cy).astype(np.int32)

    # Normalisasi Z dari [−1,+1] → [0,1] untuk keperluan warna & ukuran
    z_norm = np.clip((pts3d[:, 2] + 1.0) / 2.0, 0.0, 1.0)

    return px, py, z_norm, skala


# ═══════════════════════════════════════════════════════════════════
#  WARNA NEON DINAMIS
# ═══════════════════════════════════════════════════════════════════

def warna_neon_bgr(z_norm: float, t: float) -> tuple[int, int, int]:
    """
    Menghitung warna neon dalam format BGR berdasarkan:
      • z_norm : kedalaman (memberi variasi antar partikel)
      • t      : waktu (membuat warna berputar pelangi secara global)

    Menggunakan model HSV (Hue-Saturation-Value) agar transisi halus.
    """
    hue_deg = (t * WARNA_SHIFT_SPEED + z_norm * 220.0) % 360.0
    hsv_arr = np.uint8([[[int(hue_deg / 2), 255, 255]]])
    bgr     = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ═══════════════════════════════════════════════════════════════════
#  UTILITAS DETEKSI TANGAN
# ═══════════════════════════════════════════════════════════════════

def landmark_px(lm, lebar: int, tinggi: int) -> tuple[int, int]:
    """Konversi landmark MediaPipe (0-1) ke koordinat pixel."""
    return int(lm.x * lebar), int(lm.y * tinggi)


def posisi_telapak(tangan_lm, lebar: int, tinggi: int) -> tuple[int, int]:
    """
    Mengembalikan posisi titik tengah telapak tangan
    (rata-rata dari wrist + tengah MCP).
    """
    lm = tangan_lm.landmark
    pts = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
    ]
    xs = [lm[p].x for p in pts]
    ys = [lm[p].y for p in pts]
    return int(np.mean(xs) * lebar), int(np.mean(ys) * tinggi)


def jarak_2_tangan(h1, h2, lebar: int, tinggi: int) -> float:
    """
    Jarak Euclidean (pixel) antara wrist tangan kiri & kanan.
    """
    p1 = landmark_px(h1.landmark[mp_hands.HandLandmark.WRIST], lebar, tinggi)
    p2 = landmark_px(h2.landmark[mp_hands.HandLandmark.WRIST], lebar, tinggi)
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def jarak_ke_radius(jarak: float, lebar: int) -> float:
    """
    Peta jarak antara dua tangan (px) ke radius bola.
    Jarak referensi = 60 % lebar layar → radius maksimum.
    """
    rasio = np.clip(jarak / (lebar * 0.60), 0.0, 1.0)
    return RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * rasio


# ═══════════════════════════════════════════════════════════════════
#  RENDER PARTIKEL KE FRAME
# ═══════════════════════════════════════════════════════════════════

def render_bola(frame: np.ndarray,
                pts_rotasi: np.ndarray,
                cx: int, cy: int,
                waktu: float):
    """
    Menggambar semua partikel bola ke frame menggunakan additive blending.

    Langkah:
      1. Proyeksikan titik-titik 3D ke layar 2D.
      2. Urutkan dari belakang ke depan (Painter's Algorithm).
      3. Gambar ke layer terpisah lalu gabungkan dengan cv2.add().
    """
    H, W = frame.shape[:2]
    px, py, z_norm, _ = proyeksi(pts_rotasi, cx, cy)

    # Painter's Algorithm: gambar partikel jauh dahulu
    urutan = np.argsort(-pts_rotasi[:, 2])   # −Z → dari belakang ke depan

    layer = np.zeros_like(frame, dtype=np.uint8)

    for idx in urutan:
        x, y = int(px[idx]), int(py[idx])

        # Skip partikel di luar batas layar
        if not (2 <= x < W - 2 and 2 <= y < H - 2):
            continue

        z   = float(z_norm[idx])
        b, g, r = warna_neon_bgr(z, waktu)

        # Partikel dekat = lebih terang & besar
        intens  = 0.45 + 0.55 * (1.0 - z)
        b = int(b * intens)
        g = int(g * intens)
        r = int(r * intens)

        # Ukuran titik: partikel dekat 2-3px, jauh 1px
        ukuran = max(1, int(2.8 - z * 1.8))

        # Lingkaran glow (lebih besar, lebih redup)
        if ukuran >= 2:
            cv2.circle(layer, (x, y), ukuran + 2,
                       (b // 4, g // 4, r // 4), -1, cv2.LINE_AA)

        # Titik inti (penuh cerah)
        cv2.circle(layer, (x, y), ukuran,
                   (b, g, r), -1, cv2.LINE_AA)

    # Additive blending: partikel menambah cahaya ke frame
    cv2.add(frame, layer, dst=frame)


# ═══════════════════════════════════════════════════════════════════
#  RENDER UI OVERLAY
# ═══════════════════════════════════════════════════════════════════

def render_ui(frame: np.ndarray, fps: float,
              radius: float, n_tangan: int, mode: str):
    """Panel info semi-transparan di pojok kiri atas."""
    H, W = frame.shape[:2]

    # Latar belakang panel
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (310, 115), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    def teks(txt, y, warna=(220, 220, 220)):
        cv2.putText(frame, txt, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, warna, 1, cv2.LINE_AA)

    teks(f"FPS     : {fps:5.1f}",        32,  (80, 255, 180))
    teks(f"Radius  : {int(radius):4d} px", 56,  (80, 200, 255))
    teks(f"Tangan  : {n_tangan} / 2",      80,  (100, 255, 100))
    teks(f"Mode    : {mode}",              104, (200, 180, 255))

    # Petunjuk bawah layar
    panduan = ("2 tangan=zoom | 1 tangan=ikuti | Q=keluar")
    cv2.putText(frame, panduan, (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (140, 140, 140), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    # —— Buka webcam ——
    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        raise RuntimeError("❌ Gagal membuka webcam. Periksa koneksi kamera.")

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    kamera.set(cv2.CAP_PROP_FPS,          60)

    print("✅ Webcam aktif — jendela akan segera muncul.")
    print("🖐  Gunakan 2 tangan untuk zoom, 1 tangan untuk menggerakkan bola.")
    print("❌  Tekan Q atau ESC untuk keluar.\n")

    # —— Bangun partikel bola ternormalisasi (radius = 1) ——
    partikel_unit = buat_partikel_bola(JUMLAH_PARTIKEL)

    # —— Variabel rotasi ——
    sudut_y = 0.0
    sudut_x = 0.0

    # —— State interpolasi ——
    radius_cur  = float(RADIUS_DEFAULT)
    radius_tgt  = float(RADIUS_DEFAULT)
    pos_cur     = None   # (cx, cy) — diisi saat tangan pertama kali terdeteksi
    pos_tgt     = None

    # —— FPS counter ——
    t_prev = time.perf_counter()

    while True:
        ret, frame = kamera.read()
        if not ret:
            print("⚠️  Gagal membaca frame. Loop dihentikan.")
            break

        # Flip horizontal (mode cermin)
        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]

        # Pusat default bola
        cx_default, cy_default = W // 2, H // 2

        # —— Deteksi tangan ——
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hasil  = detektor_tangan.process(rgb)

        n_tangan  = 0
        mode_str  = "Idle"

        if hasil.multi_hand_landmarks:
            n_tangan = len(hasil.multi_hand_landmarks)

            # Gambar kerangka tangan
            for lm_tangan in hasil.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm_tangan, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 130), thickness=1, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 170, 255), thickness=1)
                )

            # ── Mode 2 tangan : kontrol zoom ──────────────────────
            if n_tangan >= 2:
                mode_str = "Zoom (2 tangan)"
                h1 = hasil.multi_hand_landmarks[0]
                h2 = hasil.multi_hand_landmarks[1]

                jarak = jarak_2_tangan(h1, h2, W, H)
                radius_tgt = jarak_ke_radius(jarak, W)

                # Gambar garis penghubung & label jarak
                p1 = landmark_px(h1.landmark[mp_hands.HandLandmark.WRIST], W, H)
                p2 = landmark_px(h2.landmark[mp_hands.HandLandmark.WRIST], W, H)
                cv2.line(frame, p1, p2, (0, 255, 200), 1, cv2.LINE_AA)
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                cv2.putText(frame, f"{int(jarak)}px", (mx, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (180, 255, 150), 1, cv2.LINE_AA)

                # Bola tetap di tengah saat mode zoom
                pos_tgt = (cx_default, cy_default)

            # ── Mode 1 tangan : bola mengikuti tangan ─────────────
            elif n_tangan == 1:
                mode_str = "Ikuti (1 tangan)"
                tx, ty = posisi_telapak(hasil.multi_hand_landmarks[0], W, H)
                pos_tgt = (tx, ty)
                # Radius tidak berubah saat mode 1 tangan

        else:
            # Tanpa tangan → bola kembali ke tengah perlahan
            pos_tgt = (cx_default, cy_default)

        # —— Inisialisasi posisi pertama kali ——
        if pos_cur is None:
            pos_cur = pos_tgt if pos_tgt else (cx_default, cy_default)
        if pos_tgt is None:
            pos_tgt = (cx_default, cy_default)

        # —— Interpolasi halus (Lerp) ——
        radius_cur += (radius_tgt - radius_cur) * LERP_RADIUS
        pos_cur = (
            pos_cur[0] + (pos_tgt[0] - pos_cur[0]) * LERP_POS,
            pos_cur[1] + (pos_tgt[1] - pos_cur[1]) * LERP_POS,
        )
        cx = int(pos_cur[0])
        cy = int(pos_cur[1])

        # —— Rotasi partikel ——
        sudut_y += SPEED_ROT_Y
        sudut_x += SPEED_ROT_X

        M_rot      = rot_y(sudut_y) @ rot_x(sudut_x)
        pts_rotasi = (M_rot @ partikel_unit.T).T   # Shape: (N, 3)
        pts_scaled = pts_rotasi * radius_cur        # Terapkan radius

        # —— Gambar partikel ke frame ——
        t_now = time.perf_counter()
        render_bola(frame, pts_scaled, cx, cy, t_now)

        # —— FPS & UI ——
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        render_ui(frame, fps, radius_cur, n_tangan, mode_str)

        # —— Tampilkan ——
        cv2.imshow("✨ Particle Sphere 3D — Kontrol Tangan", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q atau ESC
            break

    # —— Bersihkan resource ——
    kamera.release()
    cv2.destroyAllWindows()
    detektor_tangan.close()
    print("\n👋 Program selesai. Sampai jumpa!")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
