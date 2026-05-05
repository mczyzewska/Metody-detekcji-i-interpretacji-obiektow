"""
F) Przetwarzanie z siecią neuronową:
  1. Klasyfikacja (MobileNetV2 – ImageNet)
  2. Segmentacja semantyczna (DeepLab v3)
  3. Detekcja obiektów (YOLOv5 / Torch Hub)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def run_neural_network(
    pil_image: Image.Image,
    model_name: str,
    confidence: float = 0.5,
) -> tuple[Image.Image, str]:
    """
    Uruchamia wybrany model sieci neuronowej.

    Returns:
        (obraz_wynikowy, tekst_z_wynikami)
    """
    if model_name == "Klasyfikacja (MobileNetV2)":
        return _run_mobilenet(pil_image, confidence)
    elif model_name == "Segmentacja (DeepLab)":
        return _run_deeplab(pil_image)
    elif model_name == "Detekcja obiektów (YOLOv5)":
        return _run_yolo(pil_image, confidence)
    else:
        raise ValueError(f"Nieznany model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Klasyfikacja – MobileNetV2
# ──────────────────────────────────────────────────────────────────────────────

def _run_mobilenet(pil_image: Image.Image, confidence: float) -> tuple[Image.Image, str]:
    """
    Klasyfikacja obrazu przy użyciu MobileNetV2 wytrenowanego na ImageNet.
    Używa OpenCV DNN z modelem Caffe lub TensorFlow Lite.

    Fallback: prosty klasyfikator oparty na histogramie kolorów + heurystyce
    (gdy brak modelu lub sieci).
    """
    try:
        import torchvision.models as models
        import torchvision.transforms as T
        import torch

        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs[0], dim=0)
            top5 = torch.topk(probs, 5)

        # Wczytaj etykiety ImageNet
        labels = _get_imagenet_labels()
        results = []
        for prob, idx in zip(top5.values, top5.indices):
            label = labels[idx.item()] if labels else f"klasa {idx.item()}"
            results.append(f"  {label}: {prob.item()*100:.2f}%")

        result_img = _draw_classification_result(pil_image, results[:1])
        info = "MobileNetV2 – Top-5 predykcji:\n" + "\n".join(results)
        return result_img, info

    except ImportError:
        # Fallback: analiza kolorów
        return _color_analysis_fallback(pil_image)


def _color_analysis_fallback(pil_image: Image.Image) -> tuple[Image.Image, str]:
    """Prosty fallback gdy PyTorch nie jest dostępny."""
    img_np = np.array(pil_image.convert("RGB"))
    avg_color = img_np.mean(axis=(0, 1))
    r, g, b = avg_color

    dominant = "czerwony" if r > g and r > b else ("zielony" if g > r and g > b else "niebieski")
    brightness = img_np.mean()
    brightness_desc = "jasny" if brightness > 128 else "ciemny"

    result_img = _draw_classification_result(
        pil_image,
        [f"Dominujący kolor: {dominant}", f"Jasność: {brightness_desc} ({brightness:.1f})"]
    )
    info = (
        "⚠️  Tryb fallback (PyTorch niedostępny)\n\n"
        f"Analiza kolorów:\n"
        f"  Średni kolor: R={r:.1f} G={g:.1f} B={b:.1f}\n"
        f"  Dominujący: {dominant}\n"
        f"  Jasność: {brightness:.1f}/255\n\n"
        "Aby użyć MobileNetV2, zainstaluj: pip install torch torchvision"
    )
    return result_img, info


def _draw_classification_result(pil_image: Image.Image, labels: list[str]) -> Image.Image:
    """Rysuje etykiety klasyfikacji na kopii obrazu."""
    result = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    y = 10
    for label in labels:
        # Tło dla tekstu
        bbox = draw.textbbox((10, y), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((10, y), label, fill=(255, 255, 0), font=font)
        y += 35
    return result


def _get_imagenet_labels() -> list[str]:
    """Zwraca listę 1000 etykiet ImageNet (skrócona wersja)."""
    # Popularne klasy – pełna lista jest zbyt długa do hardkodowania
    # W produkcji: wczytaj z pliku imagenet_classes.txt
    try:
        import urllib.request
        import json
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Segmentacja – DeepLab v3
# ──────────────────────────────────────────────────────────────────────────────

def _run_deeplab(pil_image: Image.Image) -> tuple[Image.Image, str]:
    """
    Segmentacja semantyczna z DeepLab v3+ (backbone ResNet-50).
    Każda klasa oznaczona innym kolorem.
    """
    try:
        import torchvision.models.segmentation as seg_models
        import torchvision.transforms as T
        import torch

        model = seg_models.deeplabv3_resnet50(
            weights=seg_models.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.eval()

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)["out"][0]

        pred_mask = output.argmax(0).numpy()

        # Kolorowa mapa segmentacji
        colored = _colorize_segmentation(pred_mask)

        # Nałóż na oryginał (50% blend)
        orig_np = np.array(pil_image.convert("RGB").resize(
            (colored.shape[1], colored.shape[0])
        ))
        blended = (orig_np * 0.5 + colored * 0.5).astype(np.uint8)
        result_img = Image.fromarray(blended)

        unique_classes = np.unique(pred_mask)
        pascal_voc_classes = [
            "tło", "samolot", "rower", "ptak", "łódź", "butelka",
            "autobus", "samochód", "kot", "krzesło", "krowa",
            "stół", "pies", "koń", "motocykl", "osoba",
            "roślina doniczkowa", "owca", "sofa", "pociąg", "monitor TV"
        ]
        detected = [pascal_voc_classes[c] if c < len(pascal_voc_classes) else f"klasa {c}"
                    for c in unique_classes]

        info = f"DeepLab v3 – Segmentacja semantyczna\nWykryte klasy: {', '.join(detected)}"
        return result_img, info

    except ImportError:
        return _edge_based_segmentation_fallback(pil_image)


def _colorize_segmentation(mask: np.ndarray) -> np.ndarray:
    """Przypisuje kolory do klas segmentacji."""
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    ], dtype=np.uint8)

    colored = palette[mask % len(palette)]
    return colored


def _edge_based_segmentation_fallback(pil_image: Image.Image) -> tuple[Image.Image, str]:
    """Segmentacja oparta na krawędziach i kmeans jako fallback."""
    img_np = np.array(pil_image.convert("RGB"))
    img_float = img_np.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        img_float, 5, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img_np.shape)
    result_img = Image.fromarray(segmented)

    info = (
        "⚠️  Tryb fallback (PyTorch niedostępny)\n\n"
        "Segmentacja K-Means (5 klastrów kolorów)\n\n"
        "Aby użyć DeepLab, zainstaluj:\npip install torch torchvision"
    )
    return result_img, info


# ──────────────────────────────────────────────────────────────────────────────
# Detekcja – YOLO
# ──────────────────────────────────────────────────────────────────────────────

def _run_yolo(pil_image: Image.Image, confidence: float) -> tuple[Image.Image, str]:
    """
    Detekcja obiektów z YOLOv5 (przez torch hub).
    Fallback: detekcja twarzy z OpenCV Haar Cascades.
    """
    try:
        import torch

        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True,
                               trust_repo=True, verbose=False)
        model.conf = confidence

        results = model(pil_image)
        df = results.pandas().xyxy[0]

        result_img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(result_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            label = f"{row['name']} {row['confidence']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1, max(0, y1 - 22)), label, fill=(255, 0, 0), font=font)

        if len(df) == 0:
            info = "YOLOv5 – Nie wykryto obiektów powyżej progu pewności."
        else:
            lines = [f"  {r['name']}: {r['confidence']*100:.1f}%"
                     for _, r in df.iterrows()]
            info = f"YOLOv5 – Wykryto {len(df)} obiektów:\n" + "\n".join(lines)

        return result_img, info

    except Exception:
        return _haar_detection_fallback(pil_image)


def _haar_detection_fallback(pil_image: Image.Image) -> tuple[Image.Image, str]:
    """Detekcja twarzy za pomocą Haar Cascades jako fallback."""
    img_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))

    result_np = img_np.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_np, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result_np, "Twarz", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    n = len(faces)
    info = (
        "⚠️  Tryb fallback (YOLOv5 niedostępny) – Haar Cascades\n\n"
        f"Wykryto {n} {'twarz' if n == 1 else 'twarzy'}\n\n"
        "Aby użyć YOLOv5, zainstaluj:\npip install torch torchvision yolov5"
    )
    return result_img, info
