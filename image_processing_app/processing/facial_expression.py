"""
III-C) Wykrywanie mimiki twarzy i klasyfikacja emocji.
Backend 1: DeepFace (FER2013 model)
Backend 2: MediaPipe Face Mesh + prosty klasyfikator geometryczny
Fallback: OpenCV Haar Cascades + analiza tekstury
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOTIONS = ["szczęście", "smutek", "złość", "zaskoczenie",
            "strach", "wstręt", "neutralność"]

EMOTION_COLORS = {
    "szczęście":   (255, 215, 0),
    "smutek":      (100, 149, 237),
    "złość":       (220, 20, 60),
    "zaskoczenie": (255, 165, 0),
    "strach":      (148, 0, 211),
    "wstręt":      (0, 128, 0),
    "neutralność": (169, 169, 169),
}


def detect_facial_expression(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Wykrywa twarze i klasyfikuje emocje.

    Returns:
        (obraz_z_adnotacjami, tekst_z_wynikami)
    """
    backend = params.get("backend", "DeepFace")

    if backend == "DeepFace":
        return _deepface_detection(pil_image, params)
    else:
        return _mediapipe_expression(pil_image, params)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1: DeepFace
# ──────────────────────────────────────────────────────────────────────────────

def _deepface_detection(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """Używa biblioteki DeepFace do analizy emocji."""
    try:
        from deepface import DeepFace

        img_np = np.array(pil_image.convert("RGB"))

        # Analiza emocji (actions=['emotion'] – tylko emocje, szybciej)
        results = DeepFace.analyze(
            img_np,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        # DeepFace może zwrócić listę (wiele twarzy) lub dict
        if isinstance(results, dict):
            results = [results]

        result_img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(result_img)
        info_lines = [f"DeepFace – wykryto {len(results)} twarz(y):\n"]

        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = font_large

        for i, face_data in enumerate(results, start=1):
            emotion = face_data.get("dominant_emotion", "nieznane")
            emotion_scores = face_data.get("emotion", {})
            region = face_data.get("region", {})

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 100)
            h = region.get("h", 100)

            # Emocja po polsku
            emotion_pl = _translate_emotion(emotion)
            color = EMOTION_COLORS.get(emotion_pl, (255, 255, 255))

            if params.get("draw_faces", True):
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

            if params.get("show_score", True):
                score = emotion_scores.get(emotion, 0)
                label = f"{emotion_pl} ({score:.1f}%)"
                bbox = draw.textbbox((x, y - 28), label, font=font_large)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((x, y - 28), label, fill=color, font=font_large)

            info_lines.append(f"Twarz {i}:")
            info_lines.append(f"  Dominująca emocja: {emotion_pl}")
            info_lines.append("  Rozkład emocji:")
            for emo, score in sorted(emotion_scores.items(),
                                     key=lambda x: x[1], reverse=True):
                info_lines.append(f"    {_translate_emotion(emo)}: {score:.1f}%")
            info_lines.append("")

        return result_img, "\n".join(info_lines)

    except ImportError:
        return _opencv_face_expression_fallback(pil_image, params)
    except Exception as e:
        return _opencv_face_expression_fallback(pil_image, params)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2: MediaPipe Face Mesh
# ──────────────────────────────────────────────────────────────────────────────

def _mediapipe_expression(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """Używa MediaPipe Face Mesh do geometrycznej analizy wyrazu twarzy."""
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        img_rgb = np.array(pil_image.convert("RGB"))
        result_np = img_rgb.copy()

        info_lines = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                n_faces = len(results.multi_face_landmarks)
                info_lines.append(f"MediaPipe – wykryto {n_faces} twarz(y):\n")

                for i, face_landmarks in enumerate(results.multi_face_landmarks, start=1):
                    if params.get("draw_mesh", False):
                        mp_drawing.draw_landmarks(
                            result_np,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style()
                        )

                    # Geometryczna klasyfikacja emocji
                    emotion_pl, confidence = _geometric_emotion_classifier(
                        face_landmarks, img_rgb.shape
                    )

                    if params.get("draw_faces", True):
                        h, w = img_rgb.shape[:2]
                        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                        color = EMOTION_COLORS.get(emotion_pl, (255, 255, 255))
                        cv2.rectangle(result_np, (x1, y1), (x2, y2), color, 3)

                        if params.get("show_score", True):
                            label = f"{emotion_pl} ({confidence:.0%})"
                            cv2.putText(result_np, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    info_lines.append(f"Twarz {i}:")
                    info_lines.append(f"  Wykryta emocja: {emotion_pl}")
                    info_lines.append(f"  Pewność: {confidence:.0%}")
                    info_lines.append(f"  Liczba punktów kluczowych: {len(face_landmarks.landmark)}")
                    info_lines.append("")
            else:
                info_lines.append("❌ Nie wykryto twarzy na obrazie.")

        result_pil = Image.fromarray(result_np)
        return result_pil, "\n".join(info_lines)

    except ImportError:
        return _opencv_face_expression_fallback(pil_image, params)


def _geometric_emotion_classifier(face_landmarks, img_shape: tuple) -> tuple[str, float]:
    """
    Klasyfikuje emocję na podstawie geometrii punktów kluczowych twarzy.
    Analizuje:
      - Kąt ust (uśmiech vs smutek)
      - Szerokość oczu (zaskoczenie/strach)
      - Zmarszczenie brwi (złość/wstręt)
    """
    try:
        h, w = img_shape[:2]

        def lm(idx):
            pt = face_landmarks.landmark[idx]
            return np.array([pt.x * w, pt.y * h])

        # Punkty ust (indeksy MediaPipe Face Mesh)
        mouth_left = lm(61)
        mouth_right = lm(291)
        mouth_top = lm(13)
        mouth_bottom = lm(14)

        # Punkty oczu
        left_eye_top = lm(159)
        left_eye_bottom = lm(145)
        right_eye_top = lm(386)
        right_eye_bottom = lm(374)

        # Punkty brwi
        left_brow = lm(70)
        right_brow = lm(300)
        nose_bridge = lm(6)

        # Kąt ust (dodatni = uśmiech)
        mouth_angle = (mouth_right[1] - mouth_left[1]) / (
            np.linalg.norm(mouth_right - mouth_left) + 1e-9
        )

        # Otwarcie ust
        mouth_open = np.linalg.norm(mouth_bottom - mouth_top)
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_ratio = mouth_open / (mouth_width + 1e-9)

        # Otwarcie oczu
        left_eye_open = np.linalg.norm(left_eye_bottom - left_eye_top)
        right_eye_open = np.linalg.norm(right_eye_bottom - right_eye_top)
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        # Wysokość brwi (względem nosa)
        brow_height = (nose_bridge[1] - (left_brow[1] + right_brow[1]) / 2)

        # Klasyfikacja heurystyczna
        if mouth_angle < -0.1 and mouth_ratio < 0.3:
            return "szczęście", 0.75
        elif mouth_angle > 0.08:
            return "smutek", 0.65
        elif avg_eye_open > 18 and mouth_ratio > 0.4:
            return "zaskoczenie", 0.70
        elif brow_height < 25:
            return "złość", 0.65
        elif avg_eye_open > 20 and mouth_angle > 0.05:
            return "strach", 0.60
        else:
            return "neutralność", 0.70

    except Exception:
        return "neutralność", 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Fallback – OpenCV + analiza histogramu
# ──────────────────────────────────────────────────────────────────────────────

def _opencv_face_expression_fallback(
    pil_image: Image.Image,
    params: dict
) -> tuple[Image.Image, str]:
    """Fallback: detekcja twarzy Haar + prosta analiza."""
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))

    result_np = img_bgr.copy()
    info_lines = ["⚠️  Tryb fallback (DeepFace/MediaPipe niedostępny)\n"]

    if len(faces) == 0:
        info_lines.append("Nie wykryto twarzy.")
    else:
        info_lines.append(f"Wykryto {len(faces)} twarz(y) (Haar Cascades):\n")

        for i, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(result_np, (x, y), (x + w, y + h), (0, 255, 0), 3)

            roi_gray = gray[y:y + h, x:x + w]

            # Detekcja uśmiechu
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8,
                                                     minNeighbors=20)
            emotion = "szczęście" if len(smiles) > 0 else "neutralność"

            cv2.putText(result_np, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            info_lines.append(f"Twarz {i}: {emotion}")

    info_lines.append("\nAby użyć pełnej analizy emocji:")
    info_lines.append("  pip install deepface")
    info_lines.append("  lub")
    info_lines.append("  pip install mediapipe")

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), "\n".join(info_lines)


def _translate_emotion(emotion_en: str) -> str:
    """Tłumaczy nazwę emocji z angielskiego na polski."""
    translations = {
        "happy":    "szczęście",
        "sad":      "smutek",
        "angry":    "złość",
        "surprise": "zaskoczenie",
        "fear":     "strach",
        "disgust":  "wstręt",
        "neutral":  "neutralność",
    }
    return translations.get(emotion_en.lower(), emotion_en)
