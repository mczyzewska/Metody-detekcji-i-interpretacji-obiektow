"""
III-C) Wykrywanie mimiki twarzy i klasyfikacja emocji.

NAPRAWIONO: mediapipe ≥ 0.10 usunął mp.solutions – używamy nowego API
  Backend 1: DeepFace (najdokładniejszy – FER2013)
  Backend 2: mp.tasks.vision.FaceLandmarker (nowe API)
  Fallback:  OpenCV Haar Cascades + detekcja uśmiechu
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request

# Ścieżka do modeli MediaPipe
_MODEL_DIR       = os.path.join(os.path.dirname(__file__), "_models")
_FACE_MODEL      = os.path.join(_MODEL_DIR, "face_landmarker.task")
_FACE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/latest/"
                    "face_landmarker.task")

EMOTIONS = ["szczęście", "smutek", "złość", "zaskoczenie",
            "strach", "wstręt", "neutralność"]

EMOTION_COLORS = {
    "szczęście":   (255, 215, 0),
    "smutek":      (100, 149, 237),
    "złość":       (220, 20, 60),
    "zaskoczenie": (255, 165, 0),
    "strach":      (148, 0, 211),
    "wstręt":      (0, 160, 0),
    "neutralność": (160, 160, 160),
}

_EMOTION_EN_PL = {
    "happy":    "szczęście",
    "sad":      "smutek",
    "angry":    "złość",
    "surprise": "zaskoczenie",
    "fear":     "strach",
    "disgust":  "wstręt",
    "neutral":  "neutralność",
}


def _ensure_face_model() -> bool:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if os.path.exists(_FACE_MODEL):
        return True
    try:
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL)
        return True
    except Exception:
        return False


def _try_font(size=18):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def detect_facial_expression(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    backend = params.get("backend", "DeepFace")

    if backend == "DeepFace":
        return _deepface_detection(pil_image, params)
    else:
        return _mediapipe_new_api(pil_image, params)


# ══════════════════════════════════════════════════════════════════════════════
# Backend 1: DeepFace
# ══════════════════════════════════════════════════════════════════════════════

def _deepface_detection(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    try:
        from deepface import DeepFace
        img_np  = np.array(pil_image.convert("RGB"))
        results = DeepFace.analyze(img_np, actions=["emotion"],
                                   enforce_detection=False, silent=True)
        if isinstance(results, dict):
            results = [results]

        result_img = pil_image.copy().convert("RGB")
        draw       = ImageDraw.Draw(result_img)
        font_l     = _try_font(20)
        font_s     = _try_font(13)
        info_lines = [f"DeepFace – wykryto {len(results)} twarz(y):\n"]

        for i, fd in enumerate(results, 1):
            emotion    = fd.get("dominant_emotion", "neutral")
            scores     = fd.get("emotion", {})
            region     = fd.get("region", {})
            x, y, w, h = (region.get(k, 0) for k in ("x", "y", "w", "h"))
            if w == 0:
                w, h = pil_image.size[0] // 4, pil_image.size[1] // 4

            emotion_pl = _EMOTION_EN_PL.get(emotion.lower(), emotion)
            color      = EMOTION_COLORS.get(emotion_pl, (255, 255, 255))

            if params.get("draw_faces", True):
                draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

            if params.get("show_score", True):
                score = scores.get(emotion, 0)
                label = f"{emotion_pl} ({score:.1f}%)"
                bbox  = draw.textbbox((x, max(0, y-28)), label, font=font_l)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((x, max(0, y-28)), label, fill=color, font=font_l)

            info_lines.append(f"Twarz {i}:  {emotion_pl}")
            info_lines.append("  Rozkład emocji:")
            for emo, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                info_lines.append(f"    {_EMOTION_EN_PL.get(emo, emo)}: {sc:.1f}%")
            info_lines.append("")

        return result_img, "\n".join(info_lines)

    except ImportError:
        return _opencv_face_fallback(pil_image, params)
    except Exception:
        return _opencv_face_fallback(pil_image, params)


# ══════════════════════════════════════════════════════════════════════════════
# Backend 2: MediaPipe FaceLandmarker – NOWE API (0.10+)
# ══════════════════════════════════════════════════════════════════════════════

def _mediapipe_new_api(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    try:
        import mediapipe as mp

        if not _ensure_face_model():
            raise RuntimeError("Nie można pobrać modelu twarzy.")

        BaseOptions         = mp.tasks.BaseOptions
        FaceLandmarker      = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode         = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,  # potrzebne do emocji
        )

        img_rgb   = np.array(pil_image.convert("RGB"))
        result_np = img_rgb.copy()
        info_lines = []

        with FaceLandmarker.create_from_options(options) as landmarker:
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection = landmarker.detect(mp_image)

            if detection.face_landmarks:
                n = len(detection.face_landmarks)
                info_lines.append(f"MediaPipe FaceLandmarker – wykryto {n} twarz(y):\n")
                h_img, w_img = img_rgb.shape[:2]

                for i, (face_lms, blendshapes) in enumerate(
                        zip(detection.face_landmarks,
                            detection.face_blendshapes or [None]*n), 1):

                    # Oblicz bounding box
                    xs = [int(lm.x * w_img) for lm in face_lms]
                    ys = [int(lm.y * h_img) for lm in face_lms]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    # Emocja z blendshapes
                    emotion_pl, conf = _emotion_from_blendshapes(blendshapes)
                    color = EMOTION_COLORS.get(emotion_pl, (255, 255, 255))

                    if params.get("draw_faces", True):
                        cv2.rectangle(result_np, (x1, y1), (x2, y2), color, 3)

                    if params.get("show_score", True):
                        label = f"{emotion_pl} ({conf:.0%})"
                        cv2.putText(result_np, label, (x1, max(15, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if params.get("draw_mesh", False):
                        for lm in face_lms:
                            cx = int(lm.x * w_img)
                            cy = int(lm.y * h_img)
                            cv2.circle(result_np, (cx, cy), 1, (0, 200, 255), -1)

                    info_lines.append(f"Twarz {i}: {emotion_pl}  (pewność: {conf:.0%})")
                    info_lines.append(f"  Punkty kluczowe: {len(face_lms)}")
                    info_lines.append("")
            else:
                info_lines.append("❌ Nie wykryto twarzy na obrazie.")

        return Image.fromarray(result_np), "\n".join(info_lines)

    except ImportError:
        return _opencv_face_fallback(pil_image, params)
    except Exception:
        return _opencv_face_fallback(pil_image, params)


def _emotion_from_blendshapes(blendshapes) -> tuple[str, float]:
    """
    Mapuje blendshapes MediaPipe na emocje.
    Blendshapes to 52 współczynniki wyrazu twarzy (np. jawOpen, mouthSmileLeft itp.)
    """
    if blendshapes is None:
        return "neutralność", 0.6

    try:
        bs = {b.category_name: b.score for b in blendshapes}

        smile   = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        frown   = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        brow_d  = (bs.get("browDownLeft",   0) + bs.get("browDownRight",   0)) / 2
        brow_u  = (bs.get("browInnerUp",    0) +
                   bs.get("browOuterUpLeft",0) + bs.get("browOuterUpRight",0)) / 3
        jaw_o   = bs.get("jawOpen", 0)
        eye_sq  = (bs.get("eyeSquintLeft",  0) + bs.get("eyeSquintRight",  0)) / 2
        eye_w   = (bs.get("eyeWideLeft",    0) + bs.get("eyeWideRight",    0)) / 2
        nose_s  = bs.get("noseSneerLeft",   0)

        scores = {
            "szczęście":   smile * 0.8 + eye_sq * 0.2,
            "smutek":      frown * 0.7 + brow_u * 0.3,
            "złość":       brow_d * 0.6 + nose_s * 0.4,
            "zaskoczenie": brow_u * 0.5 + jaw_o * 0.3 + eye_w * 0.2,
            "strach":      eye_w * 0.5 + brow_u * 0.3 + jaw_o * 0.2,
            "wstręt":      nose_s * 0.7 + frown * 0.3,
            "neutralność": 0.3,
        }

        best = max(scores, key=scores.get)
        conf = min(scores[best] + 0.3, 0.99)
        return best, conf

    except Exception:
        return "neutralność", 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Fallback – OpenCV Haar Cascades
# ══════════════════════════════════════════════════════════════════════════════

def _opencv_face_fallback(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cc  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cc.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(30, 30))
    result_np = img_bgr.copy()
    info_lines = ["⚠️  Tryb fallback (DeepFace/MediaPipe niedostępny)\n"
                  "Detekcja twarzy – OpenCV Haar Cascades\n"]

    if len(faces) == 0:
        info_lines.append("Nie wykryto twarzy.")
    else:
        info_lines.append(f"Wykryto {len(faces)} twarz(y):\n")
        for i, (x, y, w, h) in enumerate(faces, 1):
            roi   = gray[y:y+h, x:x+w]
            smiles = smile_cc.detectMultiScale(roi, scaleFactor=1.8,
                                               minNeighbors=20)
            emotion = "szczęście" if len(smiles) > 0 else "neutralność"
            color   = EMOTION_COLORS.get(emotion, (0, 255, 0))
            color_bgr = (color[2], color[1], color[0])

            cv2.rectangle(result_np, (x, y), (x+w, y+h), color_bgr, 3)
            cv2.putText(result_np, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            info_lines.append(f"  Twarz {i}: {emotion}")

    info_lines += [
        "",
        "Aby uzyskać pełną analizę emocji:",
        "  pip install deepface tf-keras",
        "  lub",
        "  pip install mediapipe  (pobierze model ~30 MB automatycznie)",
    ]
    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), "\n".join(info_lines)
