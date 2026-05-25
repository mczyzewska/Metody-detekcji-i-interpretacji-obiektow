"""
III-A) Wykrywanie szkieletu człowieka w konkretnych pozach.

NAPRAWIONO: mediapipe ≥ 0.10 usunął mp.solutions – używamy nowego API
  mp.tasks.vision.PoseLandmarker  (wymaga pliku .task pobranego raz z sieci)
  Fallback: OpenCV Haar fullbody gdy brak pliku modelu lub sieci.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request

# Ścieżka do cache'owanego modelu
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "_models")
_POSE_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
_POSE_URL   = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_full/float16/latest/"
               "pose_landmarker_full.task")

# Nazwy 33 punktów kluczowych
MP_LANDMARK_NAMES = {
    0:"Nos", 1:"Lewe oko(wew)", 2:"Lewe oko", 3:"Lewe oko(zew)",
    4:"Prawe oko(wew)", 5:"Prawe oko", 6:"Prawe oko(zew)",
    7:"Lewe ucho", 8:"Prawe ucho", 9:"Usta(L)", 10:"Usta(P)",
    11:"Lewy bark", 12:"Prawy bark", 13:"Lewy łokieć", 14:"Prawy łokieć",
    15:"Lewy nadgarstek", 16:"Prawy nadgarstek",
    17:"L mały palec", 18:"P mały palec", 19:"L wskaziciel", 20:"P wskaziciel",
    21:"L kciuk", 22:"P kciuk",
    23:"Lewe biodro", 24:"Prawe biodro",
    25:"Lewe kolano", 26:"Prawe kolano",
    27:"Lewa kostka", 28:"Prawa kostka",
    29:"Lewa pięta", 30:"Prawa pięta",
    31:"Lewy paluch", 32:"Prawy paluch",
}

# Połączenia szkieletu do rysowania
SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]


def _ensure_model() -> bool:
    """Pobiera plik modelu jeśli nie istnieje. Zwraca True jeśli dostępny."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if os.path.exists(_POSE_MODEL):
        return True
    try:
        urllib.request.urlretrieve(_POSE_URL, _POSE_MODEL)
        return True
    except Exception:
        return False


def detect_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Wykrywa szkielet człowieka na obrazie PIL.
    Używa nowego mediapipe.tasks.vision.PoseLandmarker (API ≥ 0.10).
    """
    try:
        import mediapipe as mp
        # Nowe API w mediapipe 0.10+
        if not _ensure_model():
            raise RuntimeError("Nie można pobrać modelu – brak połączenia z internetem.")
        return _new_api_skeleton(pil_image, params)
    except ImportError:
        return _opencv_fallback_skeleton(pil_image, params)
    except Exception as e:
        # Jeśli nowe API zawiedzie z innego powodu – fallback
        return _opencv_fallback_skeleton(pil_image, params)


# ──────────────────────────────────────────────────────────────────────────────
# Nowe API mediapipe 0.10+ – PoseLandmarker (Tasks API)
# ──────────────────────────────────────────────────────────────────────────────

def _new_api_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    import mediapipe as mp

    det_conf  = float(params.get("det_confidence",   0.5))
    trk_conf  = float(params.get("track_confidence", 0.5))
    draw_lm   = bool(params.get("draw_landmarks",    True))
    draw_conn = bool(params.get("draw_connections",  True))

    BaseOptions     = mp.tasks.BaseOptions
    PoseLandmarker  = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL),
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=det_conf,
        min_tracking_confidence=trk_conf,
    )

    img_rgb    = np.array(pil_image.convert("RGB"))
    result_np  = img_rgb.copy()
    info_lines = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection = landmarker.detect(mp_image)

        if detection.pose_landmarks:
            n = len(detection.pose_landmarks)
            info_lines.append(f"✅ Wykryto {n} osobę/osób")

            for pose_idx, landmarks in enumerate(detection.pose_landmarks):
                h, w = img_rgb.shape[:2]

                # Piksele dla każdego punktu
                pts = {
                    i: (int(lm.x * w), int(lm.y * h))
                    for i, lm in enumerate(landmarks)
                }

                # Rysuj połączenia
                if draw_conn:
                    for (a, b) in SKELETON_CONNECTIONS:
                        if a in pts and b in pts:
                            cv2.line(result_np, pts[a], pts[b],
                                     (0, 255, 0), 2, cv2.LINE_AA)

                # Rysuj punkty kluczowe
                if draw_lm:
                    for i, (x, y) in pts.items():
                        cv2.circle(result_np, (x, y), 5, (255, 0, 0), -1)
                        cv2.circle(result_np, (x, y), 5, (255, 255, 255), 1)

                # Klasyfikuj pozę
                pose_name = _classify_pose_from_landmarks(landmarks, w, h)
                info_lines.append(f"\nOsoba {pose_idx + 1}:")
                info_lines.append(f"  Poza: {pose_name}")

                key_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                info_lines.append("  Kluczowe punkty:")
                for idx in key_ids:
                    lm   = landmarks[idx]
                    name = MP_LANDMARK_NAMES.get(idx, f"punkt {idx}")
                    info_lines.append(
                        f"    {name}: x={lm.x:.3f}, y={lm.y:.3f}, "
                        f"widoczność={lm.visibility:.2f}"
                    )

                # Napis na obrazie
                x0, y0 = pts.get(0, (10, 30))
                cv2.putText(result_np, pose_name,
                            (max(0, x0 - 40), max(20, y0 - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            info_lines.append("❌ Nie wykryto szkieletu człowieka.")
            info_lines.append("Wskazówki:")
            info_lines.append("  • Upewnij się, że cała sylwetka jest widoczna")
            info_lines.append("  • Zmniejsz próg pewności detekcji")

    result_pil = Image.fromarray(result_np)
    return result_pil, "\n".join(info_lines)


def _classify_pose_from_landmarks(landmarks, w: int, h: int) -> str:
    """Klasyfikuje pozę na podstawie kątów stawów."""
    try:
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
            return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        l_knee = angle(pt(23), pt(25), pt(27))
        r_knee = angle(pt(24), pt(26), pt(28))
        avg_knee = (l_knee + r_knee) / 2

        # pionowość tułowia: różnica Y barku i biodra
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_y      = (landmarks[23].y + landmarks[24].y) / 2
        vertical   = abs(shoulder_y - hip_y)

        if vertical < 0.15:
            return "Leżący"
        elif avg_knee > 160:
            return "Stojący"
        elif avg_knee < 115:
            return "Siedzący"
        elif avg_knee < 145:
            return "Kucający"
        else:
            return "W ruchu / pochylony"
    except Exception:
        return "Nieznana"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback – OpenCV Haar Cascades (brak mediapipe lub modelu)
# ──────────────────────────────────────────────────────────────────────────────

def _opencv_fallback_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    body_cc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml")
    upper_cc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_upperbody.xml")

    bodies = body_cc.detectMultiScale(gray, scaleFactor=1.05,
                                      minNeighbors=3, minSize=(50, 100))
    uppers = upper_cc.detectMultiScale(gray, scaleFactor=1.05,
                                       minNeighbors=3, minSize=(40, 60))

    result_np = img_bgr.copy()
    count = 0
    for (x, y, w, h) in bodies:
        cv2.rectangle(result_np, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result_np, "Sylwetka", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        count += 1
    for (x, y, w, h) in uppers:
        cv2.rectangle(result_np, (x, y), (x+w, y+h), (255, 165, 0), 2)
        cv2.putText(result_np, "Górna sylwetka", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)

    info = (
        "⚠️  Tryb fallback (MediaPipe niedostępny lub brak modelu)\n"
        "Detekcja sylwetki – OpenCV Haar Cascades\n\n"
        f"  Pełne sylwetki: {len(bodies)}\n"
        f"  Górna część:    {len(uppers)}\n\n"
        "Aby użyć pełnej detekcji szkieletu:\n"
        "  1. pip install mediapipe\n"
        "  2. Upewnij się, że masz połączenie z internetem\n"
        "     (model ~30 MB jest pobierany automatycznie przy pierwszym użyciu)"
    )
    return Image.fromarray(result_rgb), info
