"""
III-A) Wykrywanie szkieletu człowieka w konkretnych pozach.
Używa MediaPipe Pose (33 punkty kluczowe).
Fallback: OpenPose-like przy użyciu OpenCV.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Mapowanie indeksów punktów kluczowych MediaPipe → nazwy
MP_LANDMARK_NAMES = {
    0: "Nos", 1: "Lewe oko (wew.)", 2: "Lewe oko",
    3: "Lewe oko (zew.)", 4: "Prawe oko (wew.)", 5: "Prawe oko",
    6: "Prawe oko (zew.)", 7: "Lewe ucho", 8: "Prawe ucho",
    9: "Usta (lewo)", 10: "Usta (prawo)",
    11: "Lewy bark", 12: "Prawy bark",
    13: "Lewy łokieć", 14: "Prawy łokieć",
    15: "Lewy nadgarstek", 16: "Prawy nadgarstek",
    17: "Lewy mały palec", 18: "Prawy mały palec",
    19: "Lewy wskaziciel", 20: "Prawy wskaziciel",
    21: "Lewy kciuk", 22: "Prawy kciuk",
    23: "Lewe biodro", 24: "Prawe biodro",
    25: "Lewe kolano", 26: "Prawe kolano",
    27: "Lewa kostka", 28: "Prawa kostka",
    29: "Lewa pięta", 30: "Prawa pięta",
    31: "Lewy paluch", 32: "Prawy paluch",
}

# Połączenia między punktami dla rysowania szkieletu
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Progi dla klasyfikacji póz
POSE_THRESHOLDS = {
    "stojący":    {"hip_knee_angle": (160, 180), "knee_ankle_angle": (160, 180)},
    "siedzący":   {"hip_knee_angle": (80, 120)},
    "leżący":     {"vertical_ratio": (0.0, 0.3)},
}


def detect_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Wykrywa szkielet człowieka na obrazie.

    Returns:
        (obraz_z_szkieletem, tekst_z_wynikami)
    """
    try:
        import mediapipe as mp
        return _mediapipe_skeleton(pil_image, params)
    except ImportError:
        return _opencv_fallback_skeleton(pil_image, params)


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose
# ──────────────────────────────────────────────────────────────────────────────

def _mediapipe_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    det_conf = float(params.get("det_confidence", 0.5))
    trk_conf = float(params.get("track_confidence", 0.5))
    draw_lm = bool(params.get("draw_landmarks", True))
    draw_conn = bool(params.get("draw_connections", True))

    img_rgb = np.array(pil_image.convert("RGB"))
    result_img_np = img_rgb.copy()

    info_lines = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=det_conf,
        min_tracking_confidence=trk_conf,
    ) as pose:
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            if draw_conn:
                mp_drawing.draw_landmarks(
                    result_img_np,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    if draw_lm else None,
                )

            # Oblicz pozę
            landmarks = results.pose_landmarks.landmark
            pose_name = _classify_pose(landmarks, img_rgb.shape)
            info_lines.append(f"✅ Wykryto osobę")
            info_lines.append(f"🏋️  Poza: {pose_name}")
            info_lines.append("")
            info_lines.append("Kluczowe punkty (znormalizowane):")

            key_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            for idx in key_points:
                lm = landmarks[idx]
                name = MP_LANDMARK_NAMES.get(idx, f"punkt {idx}")
                info_lines.append(
                    f"  {name}: x={lm.x:.3f}, y={lm.y:.3f}, "
                    f"widoczność={lm.visibility:.2f}"
                )
        else:
            info_lines.append("❌ Nie wykryto szkieletu człowieka na obrazie.")
            info_lines.append("Wskazówki:")
            info_lines.append("  - Upewnij się, że cała sylwetka jest widoczna")
            info_lines.append("  - Zmniejsz próg pewności detekcji")

    result_pil = Image.fromarray(result_img_np)
    return result_pil, "\n".join(info_lines)


def _classify_pose(landmarks, img_shape: tuple) -> str:
    """
    Klasyfikuje pozę na podstawie kątów stawów.
    """
    try:
        h, w = img_shape[:2]

        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # Lewe biodro – kolano – kostka
        l_hip_pt, l_knee_pt, l_ankle_pt = get_pt(23), get_pt(25), get_pt(27)
        l_knee_angle = angle(l_hip_pt, l_knee_pt, l_ankle_pt)

        # Prawe biodro – kolano – kostka
        r_hip_pt, r_knee_pt, r_ankle_pt = get_pt(24), get_pt(26), get_pt(28)
        r_knee_angle = angle(r_hip_pt, r_knee_pt, r_ankle_pt)

        avg_knee = (l_knee_angle + r_knee_angle) / 2

        # Pionowość ciała
        l_shoulder = landmarks[11]
        l_hip = landmarks[23]
        vertical_diff = abs(l_shoulder.y - l_hip.y)

        if vertical_diff < 0.15:
            return "Leżący"
        elif avg_knee > 160:
            return "Stojący"
        elif avg_knee < 120:
            return "Siedzący"
        elif avg_knee < 145:
            return "Kucający"
        else:
            return "W ruchu / pochylony"

    except Exception:
        return "Nieznana"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback – prosty detektor sylwetki
# ──────────────────────────────────────────────────────────────────────────────

def _opencv_fallback_skeleton(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Fallback gdy MediaPipe jest niedostępny.
    Używa detekcji twarzy (Haar) + kontury ciała (grabcut-like).
    """
    img_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml"
    )
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    bodies = face_cascade.detectMultiScale(gray, scaleFactor=1.05,
                                           minNeighbors=3, minSize=(50, 100))

    result_np = img_np.copy()
    info_lines = [
        "⚠️  Tryb fallback (MediaPipe niedostępny)\n",
        "Detekcja sylwetki z OpenCV Haar Cascades\n",
    ]

    if len(bodies) > 0:
        for (x, y, w, h) in bodies:
            cv2.rectangle(result_np, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result_np, "Sylwetka", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        info_lines.append(f"Wykryto {len(bodies)} sylwetek")
    else:
        info_lines.append("Nie wykryto sylwetek")

    info_lines.append("\nAby użyć pełnej detekcji szkieletu:")
    info_lines.append("  pip install mediapipe")

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), "\n".join(info_lines)
