"""
C) Progowanie obrazu – 3 algorytmy:
  1. Globalne (Otsu)
  2. Adaptacyjne – Mean
  3. Adaptacyjne – Gaussian
"""

import cv2
import numpy as np
from PIL import Image


def threshold_image(pil_image: Image.Image, algorithm: str, params: dict) -> Image.Image:
    """
    Proguje obraz PIL.

    Args:
        pil_image:  Obraz wejściowy (PIL RGB).
        algorithm:  "Globalne (Otsu)" | "Adaptacyjne (Mean)" | "Adaptacyjne (Gaussian)"
        params:     słownik z parametrami

    Returns:
        Obraz po progowaniu (PIL RGB – binarny).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if algorithm == "Globalne (Otsu)":
        result = _otsu(gray)
    elif algorithm == "Adaptacyjne (Mean)":
        result = _adaptive_mean(gray, params)
    elif algorithm == "Adaptacyjne (Gaussian)":
        result = _adaptive_gaussian(gray, params)
    else:
        raise ValueError(f"Nieznany algorytm progowania: {algorithm}")

    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result_rgb)


# ── Otsu ──────────────────────────────────────────────────────────────────────

def _otsu(gray: np.ndarray) -> np.ndarray:
    """
    Metoda Otsu – automatycznie wyznacza globalny próg binaryzacji,
    minimalizując wariancję wewnątrzklasową (maksymalizuje wariancję
    międzyklasową). Działa najlepiej dla obrazów z wyraźnym bimodalnym
    histogramem.

    Przed zastosowaniem progu stosuje rozmycie Gaussowskie,
    które poprawia wyniki metody Otsu.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ── Adaptive Mean ─────────────────────────────────────────────────────────────

def _adaptive_mean(gray: np.ndarray, params: dict) -> np.ndarray:
    """
    Adaptacyjne progowanie – Mean:
    Próg dla każdego piksela obliczany jako średnia arytmetyczna
    wartości w jego otoczeniu (bloku o rozmiarze block_size × block_size)
    pomniejszona o stałą C.

    Parametry:
        block_size – rozmiar bloku sąsiedztwa (musi być nieparzyste, ≥3)
        C          – stała odejmowana od obliczonej średniej
    """
    block_size = int(params.get("block_size", 11))
    C = int(params.get("C", 2))

    # Zapewniamy nieparzysty rozmiar bloku
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    return thresh


# ── Adaptive Gaussian ─────────────────────────────────────────────────────────

def _adaptive_gaussian(gray: np.ndarray, params: dict) -> np.ndarray:
    """
    Adaptacyjne progowanie – Gaussian:
    Próg obliczany jako ważona suma wartości w sąsiedztwie
    (wagi gaussowskie – wyższe dla pikseli bliżej centrum bloku)
    pomniejszona o stałą C.

    Lepiej radzi sobie z nierównomiernym oświetleniem
    niż metoda Mean.

    Parametry:
        block_size – rozmiar bloku sąsiedztwa (nieparzysty, ≥3)
        C          – stała odejmowana od obliczonej sumy ważonej
    """
    block_size = int(params.get("block_size", 11))
    C = int(params.get("C", 2))

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    return thresh
