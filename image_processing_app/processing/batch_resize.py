"""D) Wsadowa zmiana rozdzielczości."""
import os
from PIL import Image

SUPPORTED = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
RESAMPLE  = {"LANCZOS":Image.LANCZOS,"BICUBIC":Image.BICUBIC,
             "BILINEAR":Image.BILINEAR,"NEAREST":Image.NEAREST}

def batch_resize(src, dst, params, progress_cb=None, log_cb=None):
    os.makedirs(dst, exist_ok=True)
    W,H = int(params.get("width",800)), int(params.get("height",600))
    ratio  = bool(params.get("keep_ratio",True))
    resamp = RESAMPLE.get(params.get("method","LANCZOS"), Image.LANCZOS)
    files  = [f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in SUPPORTED]
    if not files:
        if log_cb: log_cb("⚠️  Brak obsługiwanych obrazów."); return
    if log_cb: log_cb(f"Znaleziono {len(files)} obrazów…")
    for i,name in enumerate(files,1):
        try:
            with Image.open(os.path.join(src,name)) as img:
                orig = img.size
                if ratio: img.thumbnail((W,H),resamp); new=img.size
                else: img=img.resize((W,H),resamp); new=img.size
                img.save(os.path.join(dst,name), format=img.format or "PNG")
            if log_cb: log_cb(f"[{i}/{len(files)}] {name}: {orig[0]}×{orig[1]} → {new[0]}×{new[1]}")
        except Exception as e:
            if log_cb: log_cb(f"[{i}/{len(files)}] ❌ {name}: {e}")
        if progress_cb: progress_cb(int(i/len(files)*100))
