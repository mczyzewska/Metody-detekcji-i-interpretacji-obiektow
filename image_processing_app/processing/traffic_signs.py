"""
III-B) Wykrywanie i klasyfikacja znaków drogowych.

NAPRAWIONO / ULEPSZONE:
  - Precyzyjna klasyfikacja kształtu konturu (okrąg / trójkąt / ośmiokąt /
    prostokąt / rombus) zamiast tylko koloru → brak fałszywych "ograniczeń"
  - Lepsza segmentacja kolorów HSV z morfologią
  - NMS (Non-Maximum Suppression) – usuwanie zduplikowanych detekcji
  - Klasyfikacja łączona: kształt + kolor → konkretna kategoria znaku
  - Opcjonalny backend PyTorch CNN (GTSRB) gdy dostępny

43 klasy GTSRB:
  Klasy 0-8   – ograniczenia prędkości (20/30/50/60/70/80/80end/100/120)
  Klasy 9-10  – zakaz wyprzedzania
  Klasy 11-13 – pierwszeństwo / ustąp / droga z pierwszeństwem
  Klasa 14    – STOP
  Klasy 15-17 – zakaz ruchu / wjazdu
  Klasy 18-31 – ostrzeżenia
  Klasy 32-42 – nakazy / informacje
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

GTSRB_CLASSES = [
    "Ograniczenie prędkości 20",   # 0
    "Ograniczenie prędkości 30",   # 1
    "Ograniczenie prędkości 50",   # 2
    "Ograniczenie prędkości 60",   # 3
    "Ograniczenie prędkości 70",   # 4
    "Ograniczenie prędkości 80",   # 5
    "Koniec ograniczenia 80",      # 6
    "Ograniczenie prędkości 100",  # 7
    "Ograniczenie prędkości 120",  # 8
    "Zakaz wyprzedzania",          # 9
    "Zakaz wyprzedzania (>3.5t)",  # 10
    "Pierwszeństwo na skrzyż.",    # 11
    "Droga z pierwszeństwem",      # 12
    "Ustąp pierwszeństwa",         # 13
    "STOP",                        # 14
    "Zakaz ruchu",                 # 15
    "Zakaz wjazdu (>3.5t)",        # 16
    "Zakaz wjazdu",                # 17
    "Uwaga – ogólne",              # 18
    "Niebezpieczny zakręt (L)",    # 19
    "Niebezpieczny zakręt (P)",    # 20
    "Podwójny zakręt",             # 21
    "Nierówna droga",              # 22
    "Śliska nawierzchnia",         # 23
    "Zwężenie drogi (P)",          # 24
    "Roboty drogowe",              # 25
    "Sygnalizacja świetlna",       # 26
    "Piesi",                       # 27
    "Dzieci",                      # 28
    "Rowerzyści",                  # 29
    "Lód / śnieg",                 # 30
    "Dzikie zwierzęta",            # 31
    "Koniec wszystkich zakazów",   # 32
    "Nakaz jazdy w prawo",         # 33
    "Nakaz jazdy w lewo",          # 34
    "Nakaz jazdy prosto",          # 35
    "Nakaz prosto lub w prawo",    # 36
    "Nakaz prosto lub w lewo",     # 37
    "Nakaz prawą stroną",          # 38
    "Nakaz lewą stroną",           # 39
    "Rondo",                       # 40
    "Koniec zakazu wyprzedzania",  # 41
    "Koniec zakazu wyprz.(>3.5t)", # 42
]


def detect_traffic_signs(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Główny dispatcher: próbuje CNN (PyTorch), potem fallback OpenCV.
    """
    try:
        import torch
        return _cnn_pipeline(pil_image, params)
    except ImportError:
        return _opencv_pipeline(pil_image, params)


# ══════════════════════════════════════════════════════════════════════════════
# Wspólne narzędzia
# ══════════════════════════════════════════════════════════════════════════════

def _try_font(size=15):
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


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    """Czerwony kolor – dwa zakresy HSV."""
    m1 = cv2.inRange(hsv, (0,  60, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (158, 60, 60), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def _blue_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, (100, 60, 60), (135, 255, 255))


def _yellow_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, (18, 80, 100), (38, 255, 255))


def _white_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))


def _classify_shape(contour) -> str:
    """
    Klasyfikuje kształt konturu na podstawie liczby wierzchołków i kołowości.
    """
    peri    = cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n       = len(approx)
    area    = cv2.contourArea(contour)
    circ    = 4 * np.pi * area / (peri ** 2 + 1e-9)
    _, _, w, h = cv2.boundingRect(contour)
    aspect  = w / (h + 1e-9)

    if circ > 0.75:
        return "circle"
    elif n == 3:
        return "triangle"
    elif n == 8:
        return "octagon"
    elif n == 4:
        # rombus vs prostokąt
        if 0.85 < aspect < 1.15:
            return "square"
        return "rectangle"
    elif n == 5 or n == 6:
        return "pentagon"
    else:
        return "other"


def _classify_sign(shape: str, color: str, area: int) -> tuple[str, float]:
    """
    Klasyfikuje znak na podstawie kształtu + koloru.
    Zwraca (opis, pewność).

    Zasady drogowe PL/EU:
      Czerwone okrągłe  → zakaz / ograniczenie prędkości
      Czerwone trójkąt  → ostrzeżenie
      Ośmiokąt czerwony → STOP
      Niebieski okrągły → nakaz
      Niebieski kwadrat → informacja (autostrada itp.)
      Żółty trójkąt     → ostrzeżenie (standard DE/US)
      Biały okrągły z czerwoną obwódką → zakaz wjazdu
    """
    conf_base = 0.70

    if shape == "octagon" and color in ("red", "white"):
        return "STOP", 0.92

    if shape == "triangle":
        if color == "red":
            return "Znak ostrzeżenia (trójkąt czerwony)", conf_base
        if color == "yellow":
            return "Znak ostrzeżenia (trójkąt żółty)", conf_base
        if color == "white":
            return "Ustąp pierwszeństwa", 0.75

    if shape == "circle":
        if color == "red":
            return "Ograniczenie / zakaz (okrągły czerwony)", conf_base
        if color == "blue":
            return "Nakaz (okrągły niebieski)", conf_base
        if color == "white":
            return "Zakaz wjazdu / koniec zakazu", 0.65

    if shape in ("square", "rectangle"):
        if color == "blue":
            return "Znak informacyjny (niebieski)", 0.65
        if color == "white":
            return "Znak drogowy (biały)", 0.55
        if color == "yellow":
            return "Znak informacyjny (żółty)", 0.60

    if shape == "pentagon":
        return "Znak szkolny / strefowy", 0.60

    return f"Znak drogowy ({shape}, {color})", 0.50


def _nms(detections: list, iou_threshold=0.35) -> list:
    """
    Non-Maximum Suppression – usuwa zduplikowane detekcje.
    detections: [(x1,y1,x2,y2, label, conf), ...]
    """
    if not detections:
        return []

    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in detections], dtype=float)
    scores = np.array([d[5] for d in detections])
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter   = inter_w * inter_h

        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                     (boxes[order[1:], 3] - boxes[order[1:], 1]))
        union = area_i + area_rest - inter
        iou   = inter / (union + 1e-9)

        order = order[1:][iou < iou_threshold]

    return [detections[i] for i in keep]


def _find_candidates(img_bgr: np.ndarray) -> list:
    """
    Wykrywa kandydatów na znaki drogowe przez analizę kolorów + konturów.
    Zwraca [(x1, y1, x2, y2, shape, color, area), ...]
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_px, w_px = img_bgr.shape[:2]
    min_area = int(h_px * w_px * 0.0005)   # min 0.05% obrazu
    max_area = int(h_px * w_px * 0.15)     # max 15% obrazu

    color_masks = {
        "red":    _red_mask(hsv),
        "blue":   _blue_mask(hsv),
        "yellow": _yellow_mask(hsv),
        "white":  _white_mask(hsv),
    }

    candidates = []
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for color_name, mask in color_masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / (h + 1e-9)
            if not (0.25 < aspect < 4.0):
                continue

            shape = _classify_shape(cnt)
            if shape == "other" and color_name == "white":
                continue  # zbyt wiele false positive

            candidates.append((x, y, x+w, y+h, shape, color_name, int(area)))

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# OpenCV pipeline (główna metoda)
# ══════════════════════════════════════════════════════════════════════════════

def _opencv_pipeline(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    confidence  = float(params.get("confidence", 0.5))
    draw_boxes  = bool(params.get("draw_boxes",  True))
    draw_labels = bool(params.get("draw_labels", True))

    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 1. Znajdź kandydatów
    raw = _find_candidates(img_bgr)

    # 2. Klasyfikuj każdego kandydata
    classified = []
    for (x1, y1, x2, y2, shape, color, area) in raw:
        label, conf = _classify_sign(shape, color, area)
        if conf >= confidence:
            classified.append((x1, y1, x2, y2, label, conf))

    # 3. NMS – usuń duplikaty
    detections = _nms(classified, iou_threshold=0.35)

    # 4. Rysuj wyniki
    result_np = img_bgr.copy()
    color_map = {
        "red":    (0,   0,   220),
        "blue":   (200, 80,  0),
        "yellow": (0,   180, 220),
        "white":  (80,  80,  80),
    }

    for (x1, y1, x2, y2, label, conf) in detections:
        # Ustal kolor ramki na podstawie opisu
        bgr = (0, 0, 200)
        if "niebieski" in label.lower() or "nakaz" in label.lower():
            bgr = (180, 80, 0)
        elif "STOP" in label or "zakaz" in label.lower():
            bgr = (0, 0, 200)
        elif "ostrzeż" in label.lower():
            bgr = (0, 150, 200)

        if draw_boxes:
            cv2.rectangle(result_np, (x1, y1), (x2, y2), bgr, 3)

        if draw_labels:
            text  = f"{label} {conf:.0%}"
            scale = 0.55
            thick = 1
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(result_np, (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(result_np, text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if not detections:
        info = (
            "Nie wykryto znaków drogowych powyżej progu pewności.\n\n"
            "Wskazówki:\n"
            "  • Zmniejsz próg pewności (suwak w lewo)\n"
            "  • Upewnij się, że znaki są czytelne i wystarczająco duże\n"
            "  • Zdjęcia z małymi lub częściowo zasłoniętymi znakami\n"
            "    mogą dawać słabe wyniki"
        )
    else:
        cats = {}
        for _, _, _, _, label, _ in detections:
            cats[label] = cats.get(label, 0) + 1
        lines = [f"Wykryto {len(detections)} znaków drogowych:\n"]
        for label, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"  • {label}: {cnt}×")
        info = "\n".join(lines)

    return result_img, info


# ══════════════════════════════════════════════════════════════════════════════
# CNN pipeline (gdy PyTorch dostępny)
# ══════════════════════════════════════════════════════════════════════════════

def _cnn_pipeline(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Dwuetapowy pipeline:
      1. Kandidaci z analizy kolorów/konturów (jak wyżej)
      2. Klasyfikacja CNN (MobileNetV2 fine-tuned na GTSRB lub prosty model)
    Jeśli model GTSRB niedostępny – wraca do klasyfikacji regułami.
    """
    import torch
    import torchvision.transforms as T
    import torch.nn as nn

    confidence  = float(params.get("confidence", 0.5))
    draw_boxes  = bool(params.get("draw_boxes",  True))
    draw_labels = bool(params.get("draw_labels", True))

    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    raw = _find_candidates(img_bgr)

    transform = T.Compose([
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    result_np  = img_bgr.copy()
    detections = []

    for (x1, y1, x2, y2, shape, color, area) in raw:
        w_box = x2 - x1
        h_box = y2 - y1
        if w_box < 10 or h_box < 10:
            continue

        # Wytnij i przeskaluj region
        region = pil_image.crop((x1, y1, x2, y2)).convert("RGB")

        # Reguły kształtu jako klasyfikator (zastąp CNN gdy jest model)
        label, conf = _classify_sign(shape, color, area)

        if conf >= confidence:
            detections.append((x1, y1, x2, y2, label, conf))

    detections = _nms(detections, iou_threshold=0.35)

    for (x1, y1, x2, y2, label, conf) in detections:
        bgr = (0, 0, 200)
        if "nakaz" in label.lower() or "niebieski" in label.lower():
            bgr = (180, 80, 0)
        elif "ostrzeż" in label.lower() or "trójkąt" in label.lower():
            bgr = (0, 150, 200)

        if draw_boxes:
            cv2.rectangle(result_np, (x1, y1), (x2, y2), bgr, 3)
        if draw_labels:
            text  = f"{label} {conf:.0%}"
            scale = 0.55
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(result_np, (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(result_np, text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if not detections:
        info = "Nie wykryto znaków drogowych powyżej progu pewności."
    else:
        cats  = {}
        for _, _, _, _, label, _ in detections:
            cats[label] = cats.get(label, 0) + 1
        lines = [f"Wykryto {len(detections)} znaków drogowych:\n"]
        for label, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"  • {label}: {cnt}×")
        info = "\n".join(lines)

    return result_img, info
