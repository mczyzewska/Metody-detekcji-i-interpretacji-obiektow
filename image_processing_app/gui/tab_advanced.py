"""
Zakładki dla wybranych dodatkowych funkcjonalności:
A) Wykrywanie szkieletu człowieka (MediaPipe Pose)
B) Wykrywanie znaków drogowych (OpenCV + DNN / YOLO)
C) Wykrywanie mimiki twarzy (DeepFace / MediaPipe Face Mesh)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from gui.image_preview import DualPreview
from processing.skeleton_detection import detect_skeleton
from processing.traffic_signs import detect_traffic_signs
from processing.facial_expression import detect_facial_expression


def _scrollable_frame(parent):
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
# A) Szkielet człowieka
# ──────────────────────────────────────────────────────────────────────────────

class TabSkeletonDetection:
    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="A) Wykrywanie Szkieletu Człowieka",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        info = tk.Label(
            inner,
            text="Wykorzystuje MediaPipe Pose do wykrywania 33 punktów kluczowych ciała.\n"
                 "Obsługuje wykrywanie poz: stojący, siedzący, ćwiczący itp.",
            fg="#555", wraplength=600, justify=tk.CENTER
        )
        info.pack(pady=5)

        ctrl = tk.LabelFrame(inner, text="Parametry", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Min. pewność detekcji:").grid(row=0, column=0, sticky=tk.W)
        self.det_conf = tk.DoubleVar(value=0.5)
        ttk.Scale(ctrl, from_=0.1, to=1.0, variable=self.det_conf,
                  orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=8)

        tk.Label(ctrl, text="Min. pewność śledzenia:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.trk_conf = tk.DoubleVar(value=0.5)
        ttk.Scale(ctrl, from_=0.1, to=1.0, variable=self.trk_conf,
                  orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=8)

        tk.Label(ctrl, text="Rysuj punkty kluczowe:").grid(row=2, column=0, sticky=tk.W)
        self.draw_landmarks = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_landmarks).grid(row=2, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Rysuj połączenia:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.draw_connections = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_connections).grid(row=3, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶ Wykryj szkielet", command=self._run
                   ).grid(row=4, column=0, columnspan=2, pady=8)

        self.status_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.status_var, fg="#2980b9",
                 font=("Helvetica", 10)).pack(pady=4)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_text = tk.Text(inner, height=8, state=tk.DISABLED, bg="#f8f9fa")
        self.result_text.pack(fill=tk.X, padx=15, pady=4)

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return

        self.status_var.set("⏳ Wykrywanie szkieletu...")

        params = {
            "det_confidence": self.det_conf.get(),
            "track_confidence": self.trk_conf.get(),
            "draw_landmarks": self.draw_landmarks.get(),
            "draw_connections": self.draw_connections.get(),
        }

        def worker():
            try:
                result_img, info = detect_skeleton(img, params)
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
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# B) Wykrywanie znaków drogowych
# ──────────────────────────────────────────────────────────────────────────────

class TabTrafficSignDetection:
    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="B) Wykrywanie Znaków Drogowych",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        info = tk.Label(
            inner,
            text="Wykrywa i klasyfikuje znaki drogowe na obrazie.\n"
                 "Wykorzystuje model CNN trenowany na zbiorze GTSRB (43 klasy znaków).",
            fg="#555", wraplength=600, justify=tk.CENTER
        )
        info.pack(pady=5)

        ctrl = tk.LabelFrame(inner, text="Parametry", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Próg pewności:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.6)
        ttk.Scale(ctrl, from_=0.1, to=1.0, variable=self.conf_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=8)

        tk.Label(ctrl, text="Rysuj ramki:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.draw_boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_boxes).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Pokaż etykiety:").grid(row=2, column=0, sticky=tk.W)
        self.draw_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_labels).grid(row=2, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶ Wykryj znaki", command=self._run
                   ).grid(row=3, column=0, columnspan=2, pady=8)

        self.status_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.status_var, fg="#2980b9",
                 font=("Helvetica", 10)).pack(pady=4)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_text = tk.Text(inner, height=8, state=tk.DISABLED, bg="#f8f9fa")
        self.result_text.pack(fill=tk.X, padx=15, pady=4)

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return

        self.status_var.set("⏳ Wykrywanie znaków...")

        params = {
            "confidence": self.conf_var.get(),
            "draw_boxes": self.draw_boxes.get(),
            "draw_labels": self.draw_labels.get(),
        }

        def worker():
            try:
                result_img, info = detect_traffic_signs(img, params)
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
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# C) Mimika twarzy
# ──────────────────────────────────────────────────────────────────────────────

class TabFacialExpression:
    def __init__(self, parent, shared: dict):
        self.shared = shared
        _, inner = _scrollable_frame(parent)

        tk.Label(inner, text="C) Wykrywanie Mimiki Twarzy",
                 font=("Helvetica", 13, "bold")).pack(pady=10)

        info = tk.Label(
            inner,
            text="Wykrywa twarze i klasyfikuje emocje: szczęście, smutek, złość,\n"
                 "zaskoczenie, strach, wstręt, neutralność.\n"
                 "Używa DeepFace lub MediaPipe Face Mesh.",
            fg="#555", wraplength=600, justify=tk.CENTER
        )
        info.pack(pady=5)

        ctrl = tk.LabelFrame(inner, text="Parametry", padx=10, pady=8)
        ctrl.pack(padx=15, pady=5, fill=tk.X)

        tk.Label(ctrl, text="Backend:").grid(row=0, column=0, sticky=tk.W)
        self.backend_var = tk.StringVar(value="DeepFace")
        ttk.Combobox(ctrl, textvariable=self.backend_var,
                     values=["DeepFace", "MediaPipe + Custom CNN"],
                     state="readonly", width=25).grid(row=0, column=1, padx=8, pady=4)

        tk.Label(ctrl, text="Rysuj ramki twarzy:").grid(row=1, column=0, sticky=tk.W)
        self.draw_faces = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_faces).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Pokaż pewność (%):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.show_score = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.show_score).grid(row=2, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Punkty twarzy (mesh):").grid(row=3, column=0, sticky=tk.W)
        self.draw_mesh = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, variable=self.draw_mesh).grid(row=3, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶ Wykryj emocje", command=self._run
                   ).grid(row=4, column=0, columnspan=2, pady=8)

        self.status_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.status_var, fg="#2980b9",
                 font=("Helvetica", 10)).pack(pady=4)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=5)

        self.result_text = tk.Text(inner, height=8, state=tk.DISABLED, bg="#f8f9fa")
        self.result_text.pack(fill=tk.X, padx=15, pady=4)

        ttk.Button(inner, text="💾 Zapisz wynik", command=self._save).pack(pady=4)
        self._result_img = None

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return

        self.status_var.set("⏳ Analizowanie emocji...")

        params = {
            "backend": self.backend_var.get(),
            "draw_faces": self.draw_faces.get(),
            "show_score": self.show_score.get(),
            "draw_mesh": self.draw_mesh.get(),
        }

        def worker():
            try:
                result_img, info = detect_facial_expression(img, params)
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
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result_img.save(path)
            messagebox.showinfo("Sukces", f"Zapisano: {path}")
