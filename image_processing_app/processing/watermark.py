"""
E) Wsadowe dodawanie znaku wodnego do obrazów w folderze.
"""

import os
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

POSITION_MAP = {
    "Środek":       "center",
    "Lewy-górny":   "top-left",
    "Prawy-górny":  "top-right",
    "Lewy-dolny":   "bottom-left",
    "Prawy-dolny":  "bottom-right",
}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Próbuje wczytać czcionkę systemową, wraca do domyślnej jeśli brak."""
    font_candidates = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _compute_position(img_w: int, img_h: int,
                       text_w: int, text_h: int,
                       position: str, margin: int = 20) -> tuple[int, int]:
    positions = {
        "center":       ((img_w - text_w) // 2,       (img_h - text_h) // 2),
        "top-left":     (margin,                        margin),
        "top-right":    (img_w - text_w - margin,       margin),
        "bottom-left":  (margin,                        img_h - text_h - margin),
        "bottom-right": (img_w - text_w - margin,       img_h - text_h - margin),
    }
    return positions.get(position, positions["bottom-right"])


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    """Konwertuje kolor hex (#RRGGBB) do krotki RGBA."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return r, g, b, alpha


def add_watermark(img: Image.Image, params: dict) -> Image.Image:
    """
    Dodaje tekstowy znak wodny do jednego obrazu PIL.

    Args:
        img:    Obraz wejściowy (PIL).
        params: Słownik z kluczami:
                  text      – tekst znaku wodnego
                  position  – pozycja (polska nazwa)
                  alpha     – przezroczystość 0-255
                  font_size – rozmiar czcionki
                  color     – kolor hex (#RRGGBB)

    Returns:
        Obraz z nałożonym znakiem wodnym (PIL RGBA → skonwertowany do RGB).
    """
    text = params.get("text", "© Watermark")
    position_str = POSITION_MAP.get(params.get("position", "Prawy-dolny"), "bottom-right")
    alpha = int(params.get("alpha", 128))
    font_size = int(params.get("font_size", 36))
    color_hex = params.get("color", "#FFFFFF")

    # Konwersja do RGBA dla obsługi przezroczystości
    base = img.convert("RGBA")
    watermark_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)
    font = _get_font(font_size)

    # Pobierz wymiary tekstu
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Oblicz pozycję
    x, y = _compute_position(base.width, base.height, text_w, text_h, position_str)

    # Kolor z przezroczystością
    fill_color = _hex_to_rgba(color_hex, alpha)

    # Subtelny cień (ciemniejszy) dla lepszej widoczności
    shadow_alpha = min(alpha, 180)
    draw.text((x + 2, y + 2), text, font=font,
              fill=(0, 0, 0, shadow_alpha))
    draw.text((x, y), text, font=font, fill=fill_color)

    # Połącz warstwy
    combined = Image.alpha_composite(base, watermark_layer)
    return combined.convert("RGB")


def batch_watermark(
    src_folder: str,
    dst_folder: str,
    params: dict,
    progress_cb=None,
    log_cb=None,
):
    """
    Dodaje znak wodny do wszystkich obrazów w src_folder
    i zapisuje wyniki do dst_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)

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
        log_cb(f"Znaleziono {total} obrazów. Dodaję znaki wodne...")

    for i, filename in enumerate(files, start=1):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                watermarked = add_watermark(img, params)
                watermarked.save(dst_path)

            if log_cb:
                log_cb(f"[{i}/{total}] ✅ {filename}")

        except Exception as e:
            if log_cb:
                log_cb(f"[{i}/{total}] ❌ {filename}: {e}")

        if progress_cb:
            progress_cb(int(i / total * 100))
