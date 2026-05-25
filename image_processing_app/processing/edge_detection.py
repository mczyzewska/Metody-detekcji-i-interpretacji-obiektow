"""B) Wykrywanie krawędzi – Canny, Sobel, Laplacian."""
import cv2, numpy as np
from PIL import Image

def detect_edges(pil_image, algorithm, params):
    bgr  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if algorithm == "Canny":
        low, high = int(params.get("low",50)), int(params.get("high",150))
        edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), low, high)
    elif algorithm == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2+gy**2)
        edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif algorithm == "Laplacian":
        blurred = cv2.GaussianBlur(gray,(3,3),0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
