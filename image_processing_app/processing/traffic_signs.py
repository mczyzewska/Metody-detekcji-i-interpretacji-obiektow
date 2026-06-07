"""
III-B) Zakładka "Znaki drogowe" – ZMIENIONA FUNKCJA:
Zamiast detekcji znaków, konwertuje obraz na odcienie różowego.

Oferuje kilka trybów różowego:
  1. Klasyczny różowy (pink tint)
  2. Ciepły róż (warm rose)
  3. Neonowy różowy (neon pink)
  4. Pastelowy różowy (pastel)
  5. Malinowy (raspberry)
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter


PINK_MODES = [
    "Klasyczny różowy",
    "Ciepły róż",
    "Neonowy różowy",
    "Pastelowy różowy",
    "Malinowy",
]


def detect_traffic_signs(pil_image: Image.Image, params: dict) -> tuple[Image.Image, str]:
    """
    Konwertuje obraz na odcienie różowego według wybranego trybu.
    Nazwa funkcji zachowana dla kompatybilności z GUI.
    """
    mode       = params.get("pink_mode", "Klasyczny różowy")
    intensity  = float(params.get("intensity",  0.8))
    blur       = bool(params.get("blur",        False))
    vignette   = bool(params.get("vignette",    False))

    img = pil_image.convert("RGB")

    if mode == "Klasyczny różowy":
        result = _classic_pink(img, intensity)
    elif mode == "Ciepły róż":
        result = _warm_rose(img, intensity)
    elif mode == "Neonowy różowy":
        result = _neon_pink(img, intensity)
    elif mode == "Pastelowy różowy":
        result = _pastel_pink(img, intensity)
    elif mode == "Malinowy":
        result = _raspberry(img, intensity)
    else:
        result = _classic_pink(img, intensity)

    if blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=1.5))

    if vignette:
        result = _add_vignette(result)

    info = (
        f"Tryb: {mode}\n"
        f"Intensywność: {intensity:.0%}\n"
        f"Rozmycie:  {'tak' if blur else 'nie'}\n"
        f"Winietowanie: {'tak' if vignette else 'nie'}\n\n"
        f"Rozmiar obrazu: {result.size[0]}×{result.size[1]} px"
    )
    return result, info


# ── tryby różowego ─────────────────────────────────────────────────────────────

def _blend(img: Image.Image, r: int, g: int, b: int, intensity: float) -> Image.Image:
    """Blenduje obraz z jednolitym kolorem różowym."""
    arr    = np.array(img).astype(float)
    color  = np.array([[[r, g, b]]], dtype=float)
    result = arr * (1 - intensity) + color * intensity
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _classic_pink(img: Image.Image, intensity: float) -> Image.Image:
    """Klasyczny różowy – boosted R, obniżone B i G."""
    arr = np.array(img).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.0 + 60 * intensity, 0, 255)   # R +
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 - 0.35 * intensity), 0, 255) # G -
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - 0.15 * intensity) + 40 * intensity, 0, 255)  # B lekko
    return Image.fromarray(arr.astype(np.uint8))


def _warm_rose(img: Image.Image, intensity: float) -> Image.Image:
    """Ciepły różowo-kremowy odcień."""
    arr = np.array(img).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 80 * intensity, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 - 0.25 * intensity) + 20 * intensity, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - 0.40 * intensity), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _neon_pink(img: Image.Image, intensity: float) -> Image.Image:
    """Mocny neonowy różowy (#ff006e)."""
    # Desaturacja + neonowy różowy overlay
    arr  = np.array(img).astype(float)
    gray = arr.mean(axis=2, keepdims=True)
    desat = arr * (1 - intensity * 0.6) + gray * intensity * 0.6
    neon  = np.array([[[255, 0, 110]]], dtype=float)
    result = desat * (1 - intensity * 0.55) + neon * intensity * 0.55
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _pastel_pink(img: Image.Image, intensity: float) -> Image.Image:
    """Miękki pastelowy różowy – jasny, stonowany."""
    arr  = np.array(img).astype(float)
    gray = arr.mean(axis=2, keepdims=True)
    light = arr * 0.5 + gray * 0.3 + 60   # rozjaśnienie
    pastel = np.array([[[255, 182, 193]]], dtype=float)  # light pink
    result = light * (1 - intensity * 0.6) + pastel * intensity * 0.6
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _raspberry(img: Image.Image, intensity: float) -> Image.Image:
    """Malinowy – głęboki ciemny różowy."""
    arr = np.array(img).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 - 0.1 * intensity) + 135 * intensity, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 - 0.6 * intensity), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - 0.5 * intensity) + 25 * intensity, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _add_vignette(img: Image.Image) -> Image.Image:
    """Dodaje winietę (ciemne krawędzie) – wzmacnia efekt różowego."""
    arr = np.array(img).astype(float)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1 - dist * 0.65, 0.25, 1.0)
    arr *= vignette[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
