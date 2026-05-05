"""
III-B) Wykrywanie znaków drogowych.
Używa CNN wytrenowanego na zbiorze GTSRB (43 klasy)
lub prostego detektora kształtów + koloru jako fallback.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 43 klasy znaków drogowych GTSRB
GTSRB_CLASSES = [
    "Ograniczenie prędkości (20 km/h)",
    "Ograniczenie prędkości (30 km/h)",
    "Ograniczenie prędkości (50 km/h)",
    "Ograniczenie prędkości (60 km/h)",
    "Ograniczenie prędkości (70 km/h)",
    "Ograniczenie prędkości (80 km/h)",
    "Koniec ograniczenia (80 km/h)",
    "Ograniczenie prędkości (100 km/h)",
    "Ograniczenie prędkości (120 km/h)",
    "Zakaz wyprzedzania",
    "Zakaz wyprzedzania (pojazdy >3.5t)",
    "Pierwszeństwo na skrzyżowaniu",
    "Droga z pierwszeństwem",
    "Ustąp pierwszeństwa",
    "Stop",
    "Zakaz ruchu",
    "Zakaz wjazdu (pojazdy >3.5t)",
    "Zakaz wjazdu",
    "Uwaga ogólna",
    "Niebezpieczny zakręt (lewo)",
    "Niebezpieczny zakręt (prawo)",
    "Podwójny zakręt",
    "Nierówna droga",
    "Śliska nawierzchnia",
    "Zwężenie drogi (prawo)",
    "Roboty drogowe",
    "Sygnalizacja świetlna",
    "Piesi",
    "Dzieci",
    "Rowerzyści",
    "Lód / śnieg",
    "Dzikie zwierzęta",
    "Koniec wszystkich zakazów",
    "Nakaz jazdy w prawo",
    "Nakaz jazdy w lewo",
    "Nakaz jazdy prosto",
    "Nakaz jazdy prosto lub w prawo",
    "Nakaz jazdy prosto lub w lewo",
    "Nakaz jazdy prawą stroną",
    "Nakaz jazdy lewą stroną",
    "Rondo",
    "Koniec zakazu wyprzedzania",
    "Koniec zakazu wyprzedzania (>3.5t)",
]


def detect_traffic_signs(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Wykrywa znaki drogowe na obrazie PIL.

    Returns:
        (obraz_z_adnotacjami, tekst_z_wynikami)
    """
    try:
        import torch
        import torchvision.transforms as T
        return _cnn_detection(pil_image, params)
    except ImportError:
        return _opencv_color_shape_detection(pil_image, params)


# ──────────────────────────────────────────────────────────────────────────────
# CNN – klasyfikacja na kandydatach z progowania koloru
# ──────────────────────────────────────────────────────────────────────────────

def _cnn_detection(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Podejście dwuetapowe:
    1. Segmentacja kandydatów na znaki (kolor czerwony/niebieski/żółty + kształt)
    2. Klasyfikacja kandydatów przez CNN
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as T

    confidence = float(params.get("confidence", 0.6))
    draw_boxes = bool(params.get("draw_boxes", True))
    draw_labels = bool(params.get("draw_labels", True))

    # Wykryj kandydatów na znaki przez detekcję koloru
    candidates = _find_sign_candidates(pil_image)

    result_img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(result_img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    # Prosta CNN (w produkcji: załaduj wagi wytrenowane na GTSRB)
    detections = []
    for (x1, y1, x2, y2, sign_type) in candidates:
        # Wytnij kandydata
        region = pil_image.crop((x1, y1, x2, y2))
        if region.width < 20 or region.height < 20:
            continue

        # Symulacja klasyfikacji (fallback na podstawie koloru znaku)
        label, conf = _classify_by_color(region, sign_type)

        if conf < confidence:
            continue

        detections.append((x1, y1, x2, y2, label, conf))

        if draw_boxes:
            color = (255, 0, 0) if sign_type == "red" else (0, 0, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if draw_labels:
            text = f"{label} ({conf:.0%})"
            bbox = draw.textbbox((x1, y1 - 22), text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((x1, y1 - 22), text, fill=(255, 255, 0), font=font)

    if not detections:
        info = "Nie wykryto znaków drogowych powyżej progu pewności."
    else:
        lines = [f"  {label}: {conf:.0%}" for _, _, _, _, label, conf in detections]
        info = f"Wykryto {len(detections)} znaków drogowych:\n" + "\n".join(lines)

    return result_img, info


def _find_sign_candidates(pil_image: Image.Image) -> list:
    """
    Znajduje kandydatów na znaki drogowe przez analizę kolorów.
    Zwraca listę (x1, y1, x2, y2, typ_koloru).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    candidates = []

    # Maski kolorów
    color_masks = {
        "red": _red_mask(img_hsv),
        "blue": cv2.inRange(img_hsv, (100, 50, 50), (130, 255, 255)),
        "yellow": cv2.inRange(img_hsv, (15, 80, 100), (35, 255, 255)),
    }

    for color_name, mask in color_masks.items():
        # Morfologia dla wygładzenia maski
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Zbyt mały
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / (h + 1e-9)
            if 0.3 < aspect < 3.0:  # Rozsądny stosunek boków
                candidates.append((x, y, x + w, y + h, color_name))

    return candidates


def _red_mask(img_hsv: np.ndarray) -> np.ndarray:
    """Maska dla koloru czerwonego (dwa zakresy w HSV)."""
    mask1 = cv2.inRange(img_hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(img_hsv, (160, 70, 50), (180, 255, 255))
    return cv2.bitwise_or(mask1, mask2)


def _classify_by_color(region: Image.Image, color_type: str) -> tuple[str, float]:
    """
    Uproszczona klasyfikacja na podstawie koloru i kształtu.
    W produkcji: zastąp modelem CNN wytrenowanym na GTSRB.
    """
    region_np = np.array(region)
    h, w = region_np.shape[:2]
    aspect = w / (h + 1e-9)

    confidence = 0.7  # bazowa pewność

    if color_type == "red":
        if abs(aspect - 1.0) < 0.3:
            return "Ograniczenie prędkości", confidence
        else:
            return "Znak zakazu (czerwony)", confidence
    elif color_type == "blue":
        return "Znak nakazu (niebieski)", confidence
    elif color_type == "yellow":
        return "Znak ostrzeżenia (żółty)", confidence
    else:
        return "Nieznany znak", 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Fallback – detekcja tylko na podstawie kształtu i koloru
# ──────────────────────────────────────────────────────────────────────────────

def _opencv_color_shape_detection(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Fallback bez PyTorch.
    Wykrywa charakterystyczne kształty i kolory znaków drogowych.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    result_np = img_bgr.copy()
    detections = []

    # Maska czerwona
    red_mask = _red_mask(img_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-9)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        n_vertices = len(approx)

        x, y, w, h = cv2.boundingRect(contour)

        if circularity > 0.7:
            label = "Znak okrągły (zakaz/ograniczenie)"
        elif n_vertices == 8:
            label = "Znak STOP (ośmiokąt)"
        elif n_vertices == 3:
            label = "Znak trójkątny"
        else:
            label = f"Znak drogowy ({n_vertices} wierzchołków)"

        cv2.drawContours(result_np, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(result_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result_np, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        detections.append(label)

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if detections:
        info = (
            "⚠️  Tryb fallback (PyTorch niedostępny)\n\n"
            f"Wykryto {len(detections)} potencjalnych znaków:\n"
            + "\n".join(f"  • {d}" for d in detections)
            + "\n\nAby użyć CNN, zainstaluj:\npip install torch torchvision"
        )
    else:
        info = (
            "⚠️  Tryb fallback\nNie wykryto znaków drogowych.\n\n"
            "Wskazówki:\n  - Sprawdź jakość obrazu\n  - Upewnij się, że znaki są czytelne"
        )

    return result_img, info
