"""
Zakładki dla funkcjonalności podstawowych:
A) Wczytywanie i zapis obrazów
B) Wykrywanie krawędzi (3 algorytmy)
C) Progowanie (3 algorytmy)
D) Zmiana rozdzielczości wsadowa
E) Dodanie znaku wodnego wsadowe
F) Sieć neuronowa (segmentacja / klasyfikacja)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from PIL import Image, ImageDraw, ImageFont

from gui.image_preview import DualPreview, ImagePreview
from processing.edge_detection import detect_edges
from processing.thresholding import threshold_image
from processing.batch_resize import batch_resize
from processing.watermark import batch_watermark
from processing.neural_network import run_neural_network


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _scrollable_frame(parent):
    """Returns (outer_frame, inner_frame). Pack outer into parent."""
    canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(window_id, width=canvas.winfo_width())

    inner.bind("<Configure>", _on_configure)
    canvas.bind("<Configure>", _on_configure)
    return canvas, inner


# ──────────────────────────────────────────────────────────────────────────────
# A) Wczytywanie i zapis
# ──────────────────────────────────────────────────────────────────────────────

class TabLoadSave:
    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)
        self.inner = inner

        tk.Label(inner, text="A) Wczytywanie i Zapis Obrazów",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        btn_frame = tk.Frame(inner)
        btn_frame.pack(pady=5)

        ttk.Button(btn_frame, text="📂 Wczytaj obraz",
                   command=self._load, width=22).grid(row=0, column=0, padx=8, pady=4)
        ttk.Button(btn_frame, text="💾 Zapisz obraz",
                   command=self._save, width=22).grid(row=0, column=1, padx=8, pady=4)

        self.info_label = tk.Label(inner, text="Brak załadowanego obrazu",
                                   fg="#7f8c8d", font=("Helvetica", 10))
        self.info_label.pack(pady=4)

        self.preview = ImagePreview(inner, label="Podgląd", max_size=(600, 450))
        self.preview.pack(pady=8)

    def _load(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif *.webp"),
                       ("Wszystkie pliki", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.shared["set_image"](img, path)
            self.info_label.config(
                text=f"Załadowano: {os.path.basename(path)}  |  {img.size[0]}×{img.size[1]} px  |  {img.mode}",
                fg="#27ae60"
            )
            self.preview.show(img)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wczytać obrazu:\n{e}")

    def _save(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        path = filedialog.asksaveasfilename(
            title="Zapisz obraz jako",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"),
                       ("TIFF", "*.tiff"), ("Wszystkie pliki", "*.*")]
        )
        if not path:
            return
        try:
            img.save(path)
            messagebox.showinfo("Sukces", f"Obraz zapisany:\n{path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można zapisać obrazu:\n{e}")


# ──────────────────────────────────────────────────────────────────────────────
# B) Wykrywanie krawędzi
# ──────────────────────────────────────────────────────────────────────────────

class TabEdgeDetection:
    ALGORITHMS = ["Canny", "Sobel", "Laplacian"]

    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="B) Wykrywanie Krawędzi",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        ctrl = tk.LabelFrame(inner, text="Parametry", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Algorytm:").grid(row=0, column=0, sticky=tk.W)
        self.algo_var = tk.StringVar(value=self.ALGORITHMS[0])
        ttk.Combobox(ctrl, textvariable=self.algo_var,
                     values=self.ALGORITHMS, state="readonly", width=15
                     ).grid(row=0, column=1, padx=8)

        # Canny params
        tk.Label(ctrl, text="Próg dolny (Canny):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.canny_low = tk.IntVar(value=50)
        ttk.Spinbox(ctrl, from_=0, to=500, textvariable=self.canny_low, width=8
                    ).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Próg górny (Canny):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.canny_high = tk.IntVar(value=150)
        ttk.Spinbox(ctrl, from_=0, to=500, textvariable=self.canny_high, width=8
                    ).grid(row=2, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶ Wykryj krawędzie", command=self._run
                   ).grid(row=3, column=0, columnspan=2, pady=8)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_label = tk.Label(inner, text="", fg="#27ae60")
        self.result_label.pack()

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return
        algo = self.algo_var.get()
        params = {"low": self.canny_low.get(), "high": self.canny_high.get()}
        try:
            result = detect_edges(img, algo, params)
            self._result_img = result
            self.dual.show(img, result)
            self.result_label.config(text=f"Algorytm: {algo} – gotowe ✓")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def _save(self):
        if self._result_img is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję krawędzi.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# C) Progowanie
# ──────────────────────────────────────────────────────────────────────────────

class TabThresholding:
    ALGORITHMS = ["Globalne (Otsu)", "Adaptacyjne (Mean)", "Adaptacyjne (Gaussian)"]

    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="C) Progowanie Obrazu",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        ctrl = tk.LabelFrame(inner, text="Parametry", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Algorytm:").grid(row=0, column=0, sticky=tk.W)
        self.algo_var = tk.StringVar(value=self.ALGORITHMS[0])
        ttk.Combobox(ctrl, textvariable=self.algo_var,
                     values=self.ALGORITHMS, state="readonly", width=28
                     ).grid(row=0, column=1, padx=8)

        tk.Label(ctrl, text="Rozmiar bloku (adapt.):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.block_size = tk.IntVar(value=11)
        ttk.Spinbox(ctrl, from_=3, to=99, increment=2,
                    textvariable=self.block_size, width=8
                    ).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Stała C (adapt.):").grid(row=2, column=0, sticky=tk.W)
        self.const_c = tk.IntVar(value=2)
        ttk.Spinbox(ctrl, from_=0, to=50, textvariable=self.const_c, width=8
                    ).grid(row=2, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶ Proguj", command=self._run
                   ).grid(row=3, column=0, columnspan=2, pady=8)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_label = tk.Label(inner, text="", fg="#27ae60")
        self.result_label.pack()

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return
        algo = self.algo_var.get()
        params = {"block_size": self.block_size.get(), "C": self.const_c.get()}
        try:
            result = threshold_image(img, algo, params)
            self._result_img = result
            self.dual.show(img, result)
            self.result_label.config(text=f"Algorytm: {algo} – gotowe ✓")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def _save(self):
        if self._result_img is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj progowanie.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# D) Zmiana rozdzielczości wsadowa
# ──────────────────────────────────────────────────────────────────────────────

class TabResizeBatch:
    METHODS = ["LANCZOS", "BICUBIC", "NEAREST", "BILINEAR"]

    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="D) Zmiana Rozdzielczości (wsadowo)",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        ctrl = tk.LabelFrame(inner, text="Ustawienia", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        # Folder
        tk.Label(ctrl, text="Folder wejściowy:").grid(row=0, column=0, sticky=tk.W)
        self.src_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.src_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="📁", command=self._pick_src, width=3
                   ).grid(row=0, column=2)

        tk.Label(ctrl, text="Folder wyjściowy:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.dst_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.dst_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(ctrl, text="📁", command=self._pick_dst, width=3
                   ).grid(row=1, column=2)

        tk.Label(ctrl, text="Szerokość (px):").grid(row=2, column=0, sticky=tk.W)
        self.width_var = tk.IntVar(value=800)
        ttk.Spinbox(ctrl, from_=1, to=9999, textvariable=self.width_var, width=10
                    ).grid(row=2, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Wysokość (px):").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.height_var = tk.IntVar(value=600)
        ttk.Spinbox(ctrl, from_=1, to=9999, textvariable=self.height_var, width=10
                    ).grid(row=3, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Zachowaj proporcje:").grid(row=4, column=0, sticky=tk.W)
        self.keep_ratio = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.keep_ratio).grid(row=4, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Metoda interpolacji:").grid(row=5, column=0, sticky=tk.W, pady=4)
        self.method_var = tk.StringVar(value="LANCZOS")
        ttk.Combobox(ctrl, textvariable=self.method_var,
                     values=self.METHODS, state="readonly", width=15
                     ).grid(row=5, column=1, sticky=tk.W, padx=5)

        ttk.Button(ctrl, text="▶ Zmień rozdzielczość", command=self._run
                   ).grid(row=6, column=0, columnspan=3, pady=10)

        self.progress = ttk.Progressbar(inner, mode="determinate")
        self.progress.pack(fill=tk.X, padx=15, pady=4)

        self.log = tk.Text(inner, height=10, state=tk.DISABLED, bg="#f8f9fa")
        self.log.pack(fill=tk.BOTH, expand=True, padx=15, pady=4)

    def _pick_src(self):
        d = filedialog.askdirectory(title="Wybierz folder wejściowy")
        if d:
            self.src_var.set(d)

    def _pick_dst(self):
        d = filedialog.askdirectory(title="Wybierz folder wyjściowy")
        if d:
            self.dst_var.set(d)

    def _log(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _run(self):
        src = self.src_var.get()
        dst = self.dst_var.get()
        if not src or not dst:
            messagebox.showwarning("Brak folderu", "Wybierz folder wejściowy i wyjściowy.")
            return

        params = {
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "keep_ratio": self.keep_ratio.get(),
            "method": self.method_var.get(),
        }

        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)
        self.progress["value"] = 0

        def worker():
            try:
                batch_resize(src, dst, params,
                             progress_cb=lambda p: self.progress.config(value=p),
                             log_cb=self._log)
                self._log("✅ Zakończono.")
                messagebox.showinfo("Gotowe", "Zmiana rozdzielczości zakończona.")
            except Exception as e:
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# E) Znak wodny wsadowy
# ──────────────────────────────────────────────────────────────────────────────

class TabWatermark:
    POSITIONS = ["Środek", "Lewy-górny", "Prawy-górny", "Lewy-dolny", "Prawy-dolny"]

    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="E) Dodanie Znaku Wodnego (wsadowo)",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        ctrl = tk.LabelFrame(inner, text="Ustawienia", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Folder wejściowy:").grid(row=0, column=0, sticky=tk.W)
        self.src_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.src_var, width=38).grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="📁", command=self._pick_src, width=3).grid(row=0, column=2)

        tk.Label(ctrl, text="Folder wyjściowy:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.dst_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.dst_var, width=38).grid(row=1, column=1, padx=5)
        ttk.Button(ctrl, text="📁", command=self._pick_dst, width=3).grid(row=1, column=2)

        tk.Label(ctrl, text="Tekst znaku wodnego:").grid(row=2, column=0, sticky=tk.W)
        self.text_var = tk.StringVar(value="© Mój obraz")
        ttk.Entry(ctrl, textvariable=self.text_var, width=30).grid(row=2, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Pozycja:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.pos_var = tk.StringVar(value="Prawy-dolny")
        ttk.Combobox(ctrl, textvariable=self.pos_var,
                     values=self.POSITIONS, state="readonly", width=15
                     ).grid(row=3, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Przezroczystość (0-255):").grid(row=4, column=0, sticky=tk.W)
        self.alpha_var = tk.IntVar(value=128)
        ttk.Spinbox(ctrl, from_=0, to=255, textvariable=self.alpha_var, width=8
                    ).grid(row=4, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Rozmiar czcionki:").grid(row=5, column=0, sticky=tk.W, pady=4)
        self.fontsize_var = tk.IntVar(value=36)
        ttk.Spinbox(ctrl, from_=8, to=200, textvariable=self.fontsize_var, width=8
                    ).grid(row=5, column=1, sticky=tk.W, padx=5)

        tk.Label(ctrl, text="Kolor tekstu (hex):").grid(row=6, column=0, sticky=tk.W)
        self.color_var = tk.StringVar(value="#FFFFFF")
        ttk.Entry(ctrl, textvariable=self.color_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=5)

        ttk.Button(ctrl, text="▶ Dodaj znaki wodne", command=self._run
                   ).grid(row=7, column=0, columnspan=3, pady=10)

        self.progress = ttk.Progressbar(inner, mode="determinate")
        self.progress.pack(fill=tk.X, padx=15, pady=4)

        self.log = tk.Text(inner, height=10, state=tk.DISABLED, bg="#f8f9fa")
        self.log.pack(fill=tk.BOTH, expand=True, padx=15, pady=4)

    def _pick_src(self):
        d = filedialog.askdirectory()
        if d:
            self.src_var.set(d)

    def _pick_dst(self):
        d = filedialog.askdirectory()
        if d:
            self.dst_var.set(d)

    def _log(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _run(self):
        src = self.src_var.get()
        dst = self.dst_var.get()
        if not src or not dst:
            messagebox.showwarning("Brak folderu", "Wybierz folder wejściowy i wyjściowy.")
            return

        params = {
            "text": self.text_var.get(),
            "position": self.pos_var.get(),
            "alpha": self.alpha_var.get(),
            "font_size": self.fontsize_var.get(),
            "color": self.color_var.get(),
        }

        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

        def worker():
            try:
                batch_watermark(src, dst, params,
                                progress_cb=lambda p: self.progress.config(value=p),
                                log_cb=self._log)
                self._log("✅ Zakończono.")
                messagebox.showinfo("Gotowe", "Znaki wodne dodane.")
            except Exception as e:
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# F) Sieć neuronowa
# ──────────────────────────────────────────────────────────────────────────────

class TabNeuralNetwork:
    MODELS = [
        "Klasyfikacja (MobileNetV2)",
        "Segmentacja (DeepLab)",
        "Detekcja obiektów (YOLOv5)",
    ]

    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="F) Przetwarzanie z Siecią Neuronową",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        ctrl = tk.LabelFrame(inner, text="Ustawienia", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=self.MODELS[0])
        ttk.Combobox(ctrl, textvariable=self.model_var,
                     values=self.MODELS, state="readonly", width=35
                     ).grid(row=0, column=1, padx=8, pady=4)

        tk.Label(ctrl, text="Próg pewności:").grid(row=1, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Scale(ctrl, from_=0.1, to=1.0, variable=self.conf_var,
                  orient=tk.HORIZONTAL, length=200
                  ).grid(row=1, column=1, sticky=tk.W, padx=8)
        tk.Label(ctrl, textvariable=self.conf_var).grid(row=1, column=2)

        ttk.Button(ctrl, text="▶ Uruchom model", command=self._run
                   ).grid(row=2, column=0, columnspan=3, pady=8)

        self.status_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.status_var, fg="#2980b9",
                 font=("Helvetica", 10)).pack(pady=4)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_text = tk.Text(inner, height=6, state=tk.DISABLED, bg="#f8f9fa")
        self.result_text.pack(fill=tk.X, padx=15, pady=4)

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return

        self.status_var.set("⏳ Przetwarzanie...")

        def worker():
            try:
                result_img, info = run_neural_network(
                    img,
                    model_name=self.model_var.get(),
                    confidence=self.conf_var.get()
                )
                self._result_img = result_img
                self.dual.show(img, result_img)
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert(tk.END, info)
                self.result_text.config(state=tk.DISABLED)
                self.status_var.set("✅ Gotowe")
            except Exception as e:
                self.status_var.set(f"❌ Błąd: {e}")
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        if self._result_img is None:
            messagebox.showwarning("Brak wyniku", "Najpierw uruchom model.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")
