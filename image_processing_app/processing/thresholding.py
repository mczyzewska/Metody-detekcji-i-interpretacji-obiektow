"""C) Progowanie – Otsu, Adaptacyjne Mean, Adaptacyjne Gaussian."""
import cv2, numpy as np
from PIL import Image

def threshold_image(pil_image, algorithm, params):
    bgr  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if algorithm == "Globalne (Otsu)":
        blurred = cv2.GaussianBlur(gray,(5,5),0)
        _, r = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif algorithm in ("Adaptacyjne (Mean)","Adaptacyjne (Gaussian)"):
        bs = int(params.get("block_size",11))
        C  = int(params.get("C",2))
        if bs%2==0: bs+=1
        bs = max(3,bs)
        method = cv2.ADAPTIVE_THRESH_MEAN_C if "Mean" in algorithm else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        r = cv2.adaptiveThreshold(gray,255,method,cv2.THRESH_BINARY,bs,C)
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")
    return Image.fromarray(cv2.cvtColor(r, cv2.COLOR_GRAY2RGB))
