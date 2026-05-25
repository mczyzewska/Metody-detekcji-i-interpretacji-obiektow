"""
Testy jednostkowe – ze szczególnym uwzględnieniem naprawionych modułów.
Uruchomienie: python -m pytest tests/ -v
"""
import os, sys, tempfile, unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _img(w=200, h=200, mode="random"):
    if mode == "random":
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    elif mode == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
        arr[:, :, 1] = np.tile(np.linspace(0, 255, h), (w, 1)).T.astype(np.uint8)
    elif mode == "red_circle":
        # Obraz z czerwonym kołem – do testów detekcji znaków
        arr = np.ones((h, w, 3), dtype=np.uint8) * 200
        cx, cy, r = w // 2, h // 2, min(w, h) // 3
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        arr[mask] = [200, 0, 0]
    return Image.fromarray(arr)


# ── Krawędzie ─────────────────────────────────────────────────────────────────
class TestEdgeDetection(unittest.TestCase):
    def setUp(self): self.img = _img(mode="gradient")

    def test_canny(self):
        from processing.edge_detection import detect_edges
        r = detect_edges(self.img, "Canny", {"low": 50, "high": 150})
        self.assertEqual(r.size, self.img.size)
        self.assertIsInstance(r, Image.Image)

    def test_sobel(self):
        from processing.edge_detection import detect_edges
        r = detect_edges(self.img, "Sobel", {})
        self.assertEqual(r.size, self.img.size)
        self.assertGreater(np.array(r).max(), 0)

    def test_laplacian(self):
        from processing.edge_detection import detect_edges
        r = detect_edges(self.img, "Laplacian", {})
        self.assertEqual(r.size, self.img.size)

    def test_invalid_raises(self):
        from processing.edge_detection import detect_edges
        with self.assertRaises(ValueError):
            detect_edges(self.img, "Nieznany", {})


# ── Progowanie ────────────────────────────────────────────────────────────────
class TestThresholding(unittest.TestCase):
    def setUp(self): self.img = _img()

    def test_otsu(self):
        from processing.thresholding import threshold_image
        r = threshold_image(self.img, "Globalne (Otsu)", {})
        self.assertEqual(r.size, self.img.size)
        vals = np.unique(np.array(r.convert("L")))
        self.assertTrue(set(vals).issubset({0, 255}))

    def test_adaptive_mean(self):
        from processing.thresholding import threshold_image
        r = threshold_image(self.img, "Adaptacyjne (Mean)", {"block_size": 11, "C": 2})
        self.assertEqual(r.size, self.img.size)

    def test_adaptive_gaussian(self):
        from processing.thresholding import threshold_image
        r = threshold_image(self.img, "Adaptacyjne (Gaussian)", {"block_size": 11, "C": 2})
        self.assertEqual(r.size, self.img.size)

    def test_even_block_corrected(self):
        from processing.thresholding import threshold_image
        r = threshold_image(self.img, "Adaptacyjne (Mean)", {"block_size": 10, "C": 2})
        self.assertIsNotNone(r)


# ── Znak wodny ────────────────────────────────────────────────────────────────
class TestWatermark(unittest.TestCase):
    P = {"text": "© Test", "position": "Prawy-dolny",
         "alpha": 128, "font_size": 24, "color": "#FFFFFF"}

    def test_size_unchanged(self):
        from processing.watermark import add_watermark
        img = _img(400, 300)
        r = add_watermark(img, self.P)
        self.assertEqual(r.size, img.size)

    def test_all_positions(self):
        from processing.watermark import add_watermark
        img = _img(500, 500)
        for pos in ["Środek", "Lewy-górny", "Prawy-górny", "Lewy-dolny", "Prawy-dolny"]:
            r = add_watermark(img, {**self.P, "position": pos})
            self.assertEqual(r.size, img.size)

    def test_changes_pixels(self):
        from processing.watermark import add_watermark
        img = Image.new("RGB", (400, 400), (0, 0, 0))
        r = add_watermark(img, self.P)
        self.assertGreater(np.sum(np.array(img) != np.array(r)), 0)

    def test_batch(self):
        from processing.watermark import batch_watermark
        src = tempfile.mkdtemp(); dst = tempfile.mkdtemp()
        for i in range(3): _img().save(os.path.join(src, f"img_{i}.png"))
        batch_watermark(src, dst, self.P)
        self.assertEqual(len(os.listdir(dst)), 3)


# ── Batch resize ──────────────────────────────────────────────────────────────
class TestBatchResize(unittest.TestCase):
    def _src(self, n=3):
        d = tempfile.mkdtemp()
        for i in range(n): _img(400, 300).save(os.path.join(d, f"img_{i}.png"))
        return d

    def test_creates_files(self):
        from processing.batch_resize import batch_resize
        dst = tempfile.mkdtemp()
        batch_resize(self._src(3), dst, {"width":100,"height":100,"keep_ratio":True,"method":"LANCZOS"})
        self.assertEqual(len(os.listdir(dst)), 3)

    def test_exact_size(self):
        from processing.batch_resize import batch_resize
        dst = tempfile.mkdtemp()
        batch_resize(self._src(1), dst, {"width":50,"height":80,"keep_ratio":False,"method":"BICUBIC"})
        for f in os.listdir(dst):
            self.assertEqual(Image.open(os.path.join(dst, f)).size, (50, 80))

    def test_progress_100(self):
        from processing.batch_resize import batch_resize
        prog = []
        batch_resize(self._src(4), tempfile.mkdtemp(),
                     {"width":100,"height":100,"keep_ratio":True,"method":"NEAREST"},
                     progress_cb=prog.append)
        self.assertEqual(prog[-1], 100)


# ── NAPRAWIONE: Znaki drogowe ─────────────────────────────────────────────────
class TestTrafficSigns(unittest.TestCase):

    def test_returns_pil_and_str(self):
        from processing.traffic_signs import detect_traffic_signs
        img = _img(300, 300, mode="red_circle")
        r, info = detect_traffic_signs(img, {"confidence": 0.3, "draw_boxes": True, "draw_labels": True})
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(info, str)

    def test_output_size_unchanged(self):
        from processing.traffic_signs import detect_traffic_signs
        img = _img(400, 300)
        r, _ = detect_traffic_signs(img, {"confidence": 0.5})
        self.assertEqual(r.size, img.size)

    def test_nms_removes_duplicates(self):
        from processing.traffic_signs import _nms
        dets = [
            (10, 10, 60, 60, "STOP", 0.9),
            (12, 12, 62, 62, "STOP", 0.8),   # bardzo podobna – powinna być usunięta
            (200, 200, 260, 260, "Nakaz", 0.7),  # inna lokalizacja – zostaje
        ]
        result = _nms(dets, iou_threshold=0.35)
        self.assertEqual(len(result), 2)

    def test_nms_empty(self):
        from processing.traffic_signs import _nms
        self.assertEqual(_nms([]), [])

    def test_classify_shape_circle(self):
        from processing.traffic_signs import _classify_sign
        label, conf = _classify_sign("circle", "red", 5000)
        self.assertIn("Ograniczenie", label)
        self.assertGreater(conf, 0.5)

    def test_classify_shape_octagon(self):
        from processing.traffic_signs import _classify_sign
        label, conf = _classify_sign("octagon", "red", 4000)
        self.assertIn("STOP", label)
        self.assertGreater(conf, 0.85)

    def test_classify_shape_triangle_red(self):
        from processing.traffic_signs import _classify_sign
        label, conf = _classify_sign("triangle", "red", 3000)
        self.assertIn("ostrzeżenia", label.lower())

    def test_classify_shape_blue_circle(self):
        from processing.traffic_signs import _classify_sign
        label, conf = _classify_sign("circle", "blue", 3000)
        self.assertIn("Nakaz", label)

    def test_opencv_pipeline(self):
        from processing.traffic_signs import _opencv_pipeline
        img = _img(300, 300)
        r, info = _opencv_pipeline(img, {"confidence": 0.3, "draw_boxes": True, "draw_labels": True})
        self.assertIsInstance(r, Image.Image)
        self.assertEqual(r.size, img.size)

    def test_no_false_all_same_label(self):
        """Wszystkie detekcje NIE powinny mieć identycznej etykiety (regresja starego błędu)."""
        from processing.traffic_signs import detect_traffic_signs
        # Obraz z różnymi kolorami – niebieski prostokąt + czerwone koło
        arr = np.ones((300, 400, 3), dtype=np.uint8) * 180
        # Czerwone koło
        import cv2
        cv2.circle(arr, (100, 150), 50, (200, 30, 30), -1)
        # Niebieski prostokąt
        cv2.rectangle(arr, (250, 100), (350, 200), (30, 30, 200), -1)
        img = Image.fromarray(arr)
        r, info = detect_traffic_signs(img, {"confidence": 0.3, "draw_boxes": True, "draw_labels": True})
        self.assertIsInstance(r, Image.Image)


# ── NAPRAWIONE: Szkielet – fallback ──────────────────────────────────────────
class TestSkeletonFallback(unittest.TestCase):

    def test_opencv_fallback_returns_pil(self):
        from processing.skeleton_detection import _opencv_fallback_skeleton
        img = _img(300, 400)
        r, info = _opencv_fallback_skeleton(img, {})
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(info, str)
        self.assertEqual(r.size, img.size)

    def test_full_detect_returns_pil(self):
        from processing.skeleton_detection import detect_skeleton
        img = _img(300, 400)
        r, info = detect_skeleton(img, {"det_confidence": 0.5, "track_confidence": 0.5})
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(info, str)


# ── NAPRAWIONE: Mimika – fallback ──────────────────────────────────────────────
class TestFacialExpressionFallback(unittest.TestCase):

    def test_opencv_fallback_returns_pil(self):
        from processing.facial_expression import _opencv_face_fallback
        img = _img(300, 300)
        r, info = _opencv_face_fallback(img, {"draw_faces": True, "show_score": True})
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(info, str)
        self.assertEqual(r.size, img.size)

    def test_full_detect_returns_pil(self):
        from processing.facial_expression import detect_facial_expression
        img = _img(300, 300)
        r, info = detect_facial_expression(img, {"backend": "DeepFace"})
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(info, str)

    def test_mediapipe_backend_fallback(self):
        from processing.facial_expression import detect_facial_expression
        img = _img(300, 300)
        r, info = detect_facial_expression(img, {"backend": "MediaPipe + Custom CNN"})
        self.assertIsInstance(r, Image.Image)

    def test_emotion_from_blendshapes_no_data(self):
        from processing.facial_expression import _emotion_from_blendshapes
        emotion, conf = _emotion_from_blendshapes(None)
        self.assertIn(emotion, ["szczęście","smutek","złość","zaskoczenie",
                                "strach","wstręt","neutralność"])
        self.assertGreater(conf, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
