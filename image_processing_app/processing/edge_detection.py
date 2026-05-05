"""
B) Wykrywanie krawędzi – 3 algorytmy:
  1. Canny
  2. Sobel
  3. Laplacian
"""

import cv2
import numpy as np
from PIL import Image


def detect_edges(pil_image: Image.Image, algorithm: str, params: dict) -> Image.Image:
    """
    Wykrywa krawędzie w obrazie PIL.

    Args:
        pil_image:  Obraz wejściowy (PIL RGB).
        algorithm:  "Canny" | "Sobel" | "Laplacian"
        params:     słownik z parametrami

    Returns:
        Obraz z wykrytymi krawędziami (PIL RGB).
    """
    # Convert PIL → OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if algorithm == "Canny":
        edges = _canny(gray, params)
    elif algorithm == "Sobel":
        edges = _sobel(gray)
    elif algorithm == "Laplacian":
        edges = _laplacian(gray)
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")

    # Convert grayscale edges → RGB PIL
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


# ── Canny ─────────────────────────────────────────────────────────────────────

def _canny(gray: np.ndarray, params: dict) -> np.ndarray:
    """
    Algorytm Canny – wykrywanie krawędzi na podstawie gradientu obrazu
    z eliminacją szumu (Gaussian blur) i podwójnym progowaniem.

    Parametry:
        low   – dolny próg (domyślnie 50)
        high  – górny próg (domyślnie 150)
    """
    low = int(params.get("low", 50))
    high = int(params.get("high", 150))

    # Rozmycie Gaussowskie redukuje szum przed detekcją
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


# ── Sobel ─────────────────────────────────────────────────────────────────────

def _sobel(gray: np.ndarray) -> np.ndarray:
    """
    Algorytm Sobel – oblicza gradient obrazu w kierunkach X i Y,
    następnie łączy je, by uzyskać amplitudę gradientu.

    Używa jądra 3×3, operującego na wartościach typu int16,
    by uniknąć przekroczenia zakresu przy odejmowaniu pikseli.
    """
    # Gradient w kierunku X (poziomy)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Gradient w kierunku Y (pionowy)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Amplituda gradientu: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalizacja do zakresu 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


# ── Laplacian ─────────────────────────────────────────────────────────────────

def _laplacian(gray: np.ndarray) -> np.ndarray:
    """
    Algorytm Laplacian – oblicza drugą pochodną obrazu,
    wykrywając obszary szybkich zmian intensywności (krawędzie).

    Przed aplikacją filtra Laplace'a stosuje się rozmycie Gaussowskie,
    żeby zredukować wrażliwość na szum.
    """
    # Rozmycie Gaussowskie redukuje szum
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Operator Laplace'a
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Wartość bezwzględna i normalizacja
    laplacian = np.abs(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    return laplacian.astype(np.uint8)
