"""
D) Wsadowa zmiana rozdzielczości wszystkich obrazów w folderze.
"""

import os
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

INTERPOLATION_MAP = {
    "LANCZOS":  Image.LANCZOS,
    "BICUBIC":  Image.BICUBIC,
    "NEAREST":  Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
}


def batch_resize(
    src_folder: str,
    dst_folder: str,
    params: dict,
    progress_cb=None,
    log_cb=None,
):
    """
    Zmienia rozdzielczość wszystkich obsługiwanych obrazów w src_folder
    i zapisuje wyniki do dst_folder.

    Args:
        src_folder:  Ścieżka do folderu wejściowego.
        dst_folder:  Ścieżka do folderu wyjściowego.
        params:      Słownik z kluczami:
                       width      – docelowa szerokość (px)
                       height     – docelowa wysokość (px)
                       keep_ratio – zachowaj proporcje (bool)
                       method     – metoda interpolacji (str)
        progress_cb: Opcjonalne callback (float 0–100) dla paska postępu.
        log_cb:      Opcjonalne callback (str) dla logów.
    """
    os.makedirs(dst_folder, exist_ok=True)

    target_w = int(params.get("width", 800))
    target_h = int(params.get("height", 600))
    keep_ratio = bool(params.get("keep_ratio", True))
    method_name = params.get("method", "LANCZOS")
    resample = INTERPOLATION_MAP.get(method_name, Image.LANCZOS)

    # Zbierz pliki
    files = [
        f for f in os.listdir(src_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        if log_cb:
            log_cb("⚠️  Brak obsługiwanych obrazów w wybranym folderze.")
        return

    total = len(files)
    if log_cb:
        log_cb(f"Znaleziono {total} obrazów. Rozpoczynam przetwarzanie...")

    for i, filename in enumerate(files, start=1):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        try:
            with Image.open(src_path) as img:
                orig_w, orig_h = img.size

                if keep_ratio:
                    img.thumbnail((target_w, target_h), resample)
                    new_size = img.size
                else:
                    img = img.resize((target_w, target_h), resample)
                    new_size = img.size

                # Zachowaj oryginalny format jeśli możliwe
                save_format = img.format if img.format else "PNG"
                img.save(dst_path, format=save_format)

            if log_cb:
                log_cb(
                    f"[{i}/{total}] {filename}: "
                    f"{orig_w}×{orig_h} → {new_size[0]}×{new_size[1]}"
                )

        except Exception as e:
            if log_cb:
                log_cb(f"[{i}/{total}] ❌ {filename}: {e}")

        if progress_cb:
            progress_cb(int(i / total * 100))
