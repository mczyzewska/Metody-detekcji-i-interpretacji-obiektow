"""
Testy jednostkowe dla modułów przetwarzania obrazów.
Uruchomienie: python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

# Dodaj katalog projektu do ścieżki
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.edge_detection import detect_edges
from processing.thresholding import threshold_image
from processing.batch_resize import batch_resize
from processing.watermark import add_watermark, batch_watermark


def _random_img(w=200, h=200) -> Image.Image:
    """Pomocnicza funkcja – losowy obraz RGB."""
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def _gradient_img(w=200, h=200) -> Image.Image:
    """Obraz gradientowy – dobry do testów detekcji krawędzi."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
    arr[:, :, 1] = np.tile(np.linspace(0, 255, h), (w, 1)).T.astype(np.uint8)
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Testy detekcji krawędzi
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeDetection(unittest.TestCase):

    def setUp(self):
        self.img = _gradient_img()

    def test_canny_output_size(self):
        result = detect_edges(self.img, "Canny", {"low": 50, "high": 150})
        self.assertEqual(result.size, self.img.size)

    def test_sobel_output_size(self):
        result = detect_edges(self.img, "Sobel", {})
        self.assertEqual(result.size, self.img.size)

    def test_laplacian_output_size(self):
        result = detect_edges(self.img, "Laplacian", {})
        self.assertEqual(result.size, self.img.size)

    def test_canny_returns_pil_image(self):
        result = detect_edges(self.img, "Canny", {"low": 30, "high": 100})
        self.assertIsInstance(result, Image.Image)

    def test_sobel_has_edges_on_gradient(self):
        """Na obrazie gradientowym Sobel powinien znaleźć krawędzie."""
        result = detect_edges(self.img, "Sobel", {})
        arr = np.array(result)
        # Oczekujemy niezerowych pikseli (krawędzi)
        self.assertGreater(arr.max(), 0)

    def test_invalid_algorithm_raises(self):
        with self.assertRaises(ValueError):
            detect_edges(self.img, "NieistniejącyAlgorytm", {})

    def test_canny_low_threshold_default(self):
        """Canny powinien działać bez podania parametrów (użyje domyślnych)."""
        result = detect_edges(self.img, "Canny", {})
        self.assertEqual(result.size, self.img.size)


# ──────────────────────────────────────────────────────────────────────────────
# Testy progowania
# ──────────────────────────────────────────────────────────────────────────────

class TestThresholding(unittest.TestCase):

    def setUp(self):
        self.img = _random_img()

    def test_otsu_output_size(self):
        result = threshold_image(self.img, "Globalne (Otsu)", {})
        self.assertEqual(result.size, self.img.size)

    def test_adaptive_mean_output_size(self):
        result = threshold_image(self.img, "Adaptacyjne (Mean)",
                                  {"block_size": 11, "C": 2})
        self.assertEqual(result.size, self.img.size)

    def test_adaptive_gaussian_output_size(self):
        result = threshold_image(self.img, "Adaptacyjne (Gaussian)",
                                  {"block_size": 11, "C": 2})
        self.assertEqual(result.size, self.img.size)

    def test_otsu_binary_output(self):
        """Wynik Otsu powinien zawierać tylko wartości 0 i 255."""
        result = threshold_image(self.img, "Globalne (Otsu)", {})
        arr = np.array(result.convert("L"))
        unique_vals = np.unique(arr)
        self.assertTrue(set(unique_vals).issubset({0, 255}))

    def test_adaptive_even_block_size_corrected(self):
        """Parzysty block_size powinien zostać skorygowany do nieparzystego."""
        # Nie powinno rzucać wyjątku
        result = threshold_image(self.img, "Adaptacyjne (Mean)",
                                  {"block_size": 10, "C": 2})
        self.assertIsNotNone(result)

    def test_invalid_algorithm_raises(self):
        with self.assertRaises(ValueError):
            threshold_image(self.img, "NieistniejącyAlgorytm", {})


# ──────────────────────────────────────────────────────────────────────────────
# Testy wsadowej zmiany rozdzielczości
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchResize(unittest.TestCase):

    def _make_src_folder(self, n=3, fmt="PNG"):
        """Tworzy tymczasowy folder z n obrazami."""
        tmpdir = tempfile.mkdtemp()
        for i in range(n):
            img = _random_img(400, 300)
            ext = ".png" if fmt == "PNG" else ".jpg"
            img.save(os.path.join(tmpdir, f"img_{i}{ext}"))
        return tmpdir

    def test_creates_output_files(self):
        src = self._make_src_folder(3)
        dst = tempfile.mkdtemp()
        batch_resize(src, dst, {"width": 100, "height": 100,
                                "keep_ratio": True, "method": "LANCZOS"})
        self.assertEqual(len(os.listdir(dst)), 3)

    def test_output_size_no_ratio(self):
        src = self._make_src_folder(1)
        dst = tempfile.mkdtemp()
        batch_resize(src, dst, {"width": 50, "height": 80,
                                "keep_ratio": False, "method": "BICUBIC"})
        for f in os.listdir(dst):
            img = Image.open(os.path.join(dst, f))
            self.assertEqual(img.size, (50, 80))

    def test_output_size_with_ratio(self):
        src = self._make_src_folder(1)
        dst = tempfile.mkdtemp()
        batch_resize(src, dst, {"width": 100, "height": 100,
                                "keep_ratio": True, "method": "LANCZOS"})
        for f in os.listdir(dst):
            img = Image.open(os.path.join(dst, f))
            # Przy zachowaniu proporcji żaden wymiar nie przekracza target
            self.assertLessEqual(img.width, 100)
            self.assertLessEqual(img.height, 100)

    def test_empty_folder(self):
        src = tempfile.mkdtemp()
        dst = tempfile.mkdtemp()
        logs = []
        batch_resize(src, dst, {"width": 100, "height": 100,
                                "keep_ratio": True, "method": "LANCZOS"},
                     log_cb=logs.append)
        self.assertTrue(any("Brak" in l for l in logs))

    def test_progress_callback(self):
        src = self._make_src_folder(4)
        dst = tempfile.mkdtemp()
        progress_values = []
        batch_resize(src, dst, {"width": 100, "height": 100,
                                "keep_ratio": True, "method": "LANCZOS"},
                     progress_cb=progress_values.append)
        self.assertEqual(progress_values[-1], 100)

    def test_creates_dst_if_not_exists(self):
        src = self._make_src_folder(1)
        dst = os.path.join(tempfile.mkdtemp(), "new_folder", "nested")
        batch_resize(src, dst, {"width": 100, "height": 100,
                                "keep_ratio": True, "method": "NEAREST"})
        self.assertTrue(os.path.isdir(dst))


# ──────────────────────────────────────────────────────────────────────────────
# Testy znaku wodnego
# ──────────────────────────────────────────────────────────────────────────────

class TestWatermark(unittest.TestCase):

    BASE_PARAMS = {
        "text": "© Test 2026",
        "position": "Prawy-dolny",
        "alpha": 128,
        "font_size": 24,
        "color": "#FFFFFF",
    }

    def test_returns_pil_image(self):
        img = _random_img()
        result = add_watermark(img, self.BASE_PARAMS)
        self.assertIsInstance(result, Image.Image)

    def test_output_size_unchanged(self):
        img = _random_img(400, 300)
        result = add_watermark(img, self.BASE_PARAMS)
        self.assertEqual(result.size, img.size)

    def test_all_positions(self):
        positions = ["Środek", "Lewy-górny", "Prawy-górny",
                     "Lewy-dolny", "Prawy-dolny"]
        img = _random_img(500, 500)
        for pos in positions:
            params = {**self.BASE_PARAMS, "position": pos}
            result = add_watermark(img, params)
            self.assertEqual(result.size, img.size, f"Błąd dla pozycji: {pos}")

    def test_watermark_changes_pixels(self):
        """Znak wodny powinien zmienić przynajmniej część pikseli."""
        img = Image.new("RGB", (400, 400), color=(0, 0, 0))
        result = add_watermark(img, self.BASE_PARAMS)
        arr_orig = np.array(img)
        arr_result = np.array(result)
        diff = np.sum(arr_orig != arr_result)
        self.assertGreater(diff, 0)

    def test_transparent_watermark(self):
        """Alpha=0 – znak wodny prawie niewidoczny, ale obraz niezmieniony rozmiarem."""
        img = _random_img()
        params = {**self.BASE_PARAMS, "alpha": 0}
        result = add_watermark(img, params)
        self.assertEqual(result.size, img.size)

    def test_batch_watermark_creates_files(self):
        src = tempfile.mkdtemp()
        dst = tempfile.mkdtemp()
        for i in range(3):
            _random_img().save(os.path.join(src, f"img_{i}.png"))

        batch_watermark(src, dst, self.BASE_PARAMS)
        self.assertEqual(len(os.listdir(dst)), 3)

    def test_batch_watermark_progress(self):
        src = tempfile.mkdtemp()
        dst = tempfile.mkdtemp()
        for i in range(5):
            _random_img().save(os.path.join(src, f"img_{i}.png"))

        progress = []
        batch_watermark(src, dst, self.BASE_PARAMS, progress_cb=progress.append)
        self.assertEqual(progress[-1], 100)

    def test_hex_color_variants(self):
        img = _random_img()
        for color in ["#FF0000", "#00FF00", "#0000FF", "#FFF", "#000"]:
            params = {**self.BASE_PARAMS, "color": color}
            result = add_watermark(img, params)
            self.assertIsNotNone(result)


# ──────────────────────────────────────────────────────────────────────────────
# Testy modułów zaawansowanych (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class TestAdvancedFallbacks(unittest.TestCase):

    def setUp(self):
        self.img = _random_img(300, 300)

    def test_skeleton_fallback(self):
        from processing.skeleton_detection import _opencv_fallback_skeleton
        result, info = _opencv_fallback_skeleton(self.img, {})
        self.assertIsInstance(result, Image.Image)
        self.assertIsInstance(info, str)
        self.assertEqual(result.size, self.img.size)

    def test_traffic_signs_fallback(self):
        from processing.traffic_signs import _opencv_color_shape_detection
        result, info = _opencv_color_shape_detection(
            self.img,
            {"confidence": 0.5, "draw_boxes": True, "draw_labels": True}
        )
        self.assertIsInstance(result, Image.Image)
        self.assertIsInstance(info, str)

    def test_facial_expression_fallback(self):
        from processing.facial_expression import _opencv_face_expression_fallback
        result, info = _opencv_face_expression_fallback(
            self.img,
            {"draw_faces": True, "show_score": True}
        )
        self.assertIsInstance(result, Image.Image)
        self.assertIsInstance(info, str)

    def test_neural_network_color_fallback(self):
        from processing.neural_network import _color_analysis_fallback
        result, info = _color_analysis_fallback(self.img)
        self.assertIsInstance(result, Image.Image)
        self.assertIn("Analiza kolorów", info)

    def test_neural_network_haar_fallback(self):
        from processing.neural_network import _haar_detection_fallback
        result, info = _haar_detection_fallback(self.img)
        self.assertIsInstance(result, Image.Image)

    def test_neural_network_kmeans_fallback(self):
        from processing.neural_network import _edge_based_segmentation_fallback
        result, info = _edge_based_segmentation_fallback(self.img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
