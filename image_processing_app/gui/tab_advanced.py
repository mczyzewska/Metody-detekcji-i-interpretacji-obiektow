"""
Zakładki zaawansowane – naprawione:
  III-A) Szkielet człowieka  – nowe API mediapipe 0.10+
  III-B) Znaki drogowe       – lepsza klasyfikacja kształt+kolor+NMS
  III-C) Mimika twarzy       – nowe API mediapipe 0.10+
"""
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from gui.image_preview import DualPreview
from processing.skeleton_detection import detect_skeleton
from processing.traffic_signs import detect_traffic_signs
from processing.facial_expression import detect_facial_expression


def _scroll(parent):
    c = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient="vertical", command=c.yview)
    c.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(c)
    wid = c.create_window((0, 0), window=inner, anchor="nw")

    def _cfg(e):
        c.configure(scrollregion=c.bbox("all"))
        c.itemconfig(wid, width=c.winfo_width())

    inner.bind("<Configure>", _cfg)
    c.bind("<Configure>", _cfg)
    return inner


def _sec(parent, title):
    f = tk.LabelFrame(parent, text=title, padx=10, pady=8,
                      font=("Helvetica", 10, "bold"))
    f.pack(padx=14, pady=6, fill=tk.X)
    return f


def _result_box(parent):
    """Textbox na wyniki z scrollbarem."""
    rf = tk.LabelFrame(parent, text="Wyniki", padx=8, pady=6)
    rf.pack(fill=tk.X, padx=14, pady=4)
    sb = ttk.Scrollbar(rf)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    t = tk.Text(rf, height=8, state=tk.DISABLED,
                bg="#f4f6f7", yscrollcommand=sb.set)
    t.pack(fill=tk.BOTH)
    sb.config(command=t.yview)
    return t


def _set_result(textbox, text):
    textbox.config(state=tk.NORMAL)
    textbox.delete("1.0", tk.END)
    textbox.insert(tk.END, text)
    textbox.config(state=tk.DISABLED)


# ══════════════════════════════════════════════════════════════════════════════
# III-A) Szkielet człowieka
# ══════════════════════════════════════════════════════════════════════════════

class TabSkeletonDetection:
    def __init__(self, parent, shared):
        self.shared = shared
        self._result = None
        inner = _scroll(parent)

        tk.Label(inner, text="A) Wykrywanie Szkieletu Człowieka",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        tk.Label(
            inner,
            text="Używa MediaPipe PoseLandmarker (Tasks API – mediapipe ≥ 0.10).\n"
                 "Wykrywa 33 punkty kluczowe ciała i klasyfikuje pozę.\n"
                 "Model (~30 MB) jest pobierany automatycznie przy pierwszym użyciu.",
            fg="#555", justify=tk.CENTER, wraplength=650
        ).pack(pady=4)

        ctrl = _sec(inner, "Parametry")

        tk.Label(ctrl, text="Min. pewność detekcji:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.det = tk.DoubleVar(value=0.5)
        frm0 = tk.Frame(ctrl); frm0.grid(row=0, column=1, sticky=tk.W, padx=8)
        ttk.Scale(frm0, from_=0.1, to=1.0, variable=self.det,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        tk.Label(frm0, textvariable=self.det, width=5).pack(side=tk.LEFT)

        tk.Label(ctrl, text="Min. pewność śledzenia:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.trk = tk.DoubleVar(value=0.5)
        frm1 = tk.Frame(ctrl); frm1.grid(row=1, column=1, sticky=tk.W, padx=8)
        ttk.Scale(frm1, from_=0.1, to=1.0, variable=self.trk,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        tk.Label(frm1, textvariable=self.trk, width=5).pack(side=tk.LEFT)

        tk.Label(ctrl, text="Rysuj punkty kluczowe:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.draw_lm = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_lm).grid(row=2, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Rysuj połączenia szkieletu:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.draw_cn = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_cn).grid(row=3, column=1, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="▶  Wykryj szkielet",
                   command=self._run).grid(row=4, column=0, columnspan=2, pady=10)

        self.status = tk.Label(inner, text="", fg="#2980b9", font=("Helvetica", 10))
        self.status.pack(pady=2)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=6)

        self.rt = _result_box(inner)
        ttk.Button(inner, text="💾  Zapisz wynik", command=self._save).pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return
        self.status.config(text="⏳  Wykrywanie szkieletu…", fg="#e67e22")

        params = {
            "det_confidence":   round(self.det.get(), 2),
            "track_confidence": round(self.trk.get(), 2),
            "draw_landmarks":   self.draw_lm.get(),
            "draw_connections": self.draw_cn.get(),
        }

        def worker():
            try:
                res, info = detect_skeleton(img, params)
                self._result = res
                self.dual.show(img, res)
                _set_result(self.rt, info)
                self.status.config(text="✅  Zakończono pomyślnie", fg="#27ae60")
            except Exception as e:
                self.status.config(text=f"❌  Błąd: {e}", fg="#e74c3c")
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        if self._result is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result.save(path)
            messagebox.showinfo("Zapisano", path)


# ══════════════════════════════════════════════════════════════════════════════
# III-B) Znaki drogowe
# ══════════════════════════════════════════════════════════════════════════════

class TabTrafficSignDetection:
    def __init__(self, parent, shared):
        self.shared = shared
        self._result = None
        inner = _scroll(parent)

        tk.Label(inner, text="B) Wykrywanie Znaków Drogowych",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        tk.Label(
            inner,
            text="Detekcja dwuetapowa: segmentacja kandydatów (kolor HSV + kontury)\n"
                 "→ klasyfikacja przez kształt konturu (okrąg/trójkąt/ośmiokąt/prostokąt)\n"
                 "→ NMS (usuwanie duplikatów) → etykietowanie kategorii znaku.\n"
                 "43 klasy GTSRB. Jeśli PyTorch dostępny – używa CNN.",
            fg="#555", justify=tk.CENTER, wraplength=650
        ).pack(pady=4)

        ctrl = _sec(inner, "Parametry")

        tk.Label(ctrl, text="Próg pewności:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.conf = tk.DoubleVar(value=0.50)
        frm = tk.Frame(ctrl); frm.grid(row=0, column=1, sticky=tk.W, padx=8)
        self._conf_scale = ttk.Scale(frm, from_=0.1, to=0.95, variable=self.conf,
                                     orient=tk.HORIZONTAL, length=220)
        self._conf_scale.pack(side=tk.LEFT)
        self._conf_lbl = tk.Label(frm, width=5)
        self._conf_lbl.pack(side=tk.LEFT, padx=4)
        self.conf.trace_add("write", lambda *_: self._conf_lbl.config(
            text=f"{self.conf.get():.2f}"))
        self._conf_lbl.config(text="0.50")

        tk.Label(ctrl, text="Rysuj ramki:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.draw_boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_boxes).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Pokaż etykiety:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.draw_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_labels).grid(row=2, column=1, sticky=tk.W, padx=8)

        # Wskazówka dla użytkownika
        hint = tk.LabelFrame(ctrl, text="💡 Wskazówka", padx=8, pady=4)
        hint.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=6)
        tk.Label(
            hint,
            text="Jeśli wykrywa za mało znaków: przesuń próg pewności w lewo (np. 0.40).\n"
                 "Jeśli wykrywa za dużo fałszywych: przesuń próg w prawo (np. 0.70).",
            fg="#555", font=("Helvetica", 9), justify=tk.LEFT
        ).pack(anchor=tk.W)

        ttk.Button(ctrl, text="▶  Wykryj znaki",
                   command=self._run).grid(row=4, column=0, columnspan=3, pady=10)

        self.status = tk.Label(inner, text="", fg="#2980b9", font=("Helvetica", 10))
        self.status.pack(pady=2)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=6)

        self.rt = _result_box(inner)
        ttk.Button(inner, text="💾  Zapisz wynik", command=self._save).pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return
        self.status.config(text="⏳  Wykrywanie znaków…", fg="#e67e22")

        params = {
            "confidence":   round(self.conf.get(), 2),
            "draw_boxes":   self.draw_boxes.get(),
            "draw_labels":  self.draw_labels.get(),
        }

        def worker():
            try:
                res, info = detect_traffic_signs(img, params)
                self._result = res
                self.dual.show(img, res)
                _set_result(self.rt, info)
                self.status.config(text="✅  Zakończono pomyślnie", fg="#27ae60")
            except Exception as e:
                self.status.config(text=f"❌  Błąd: {e}", fg="#e74c3c")
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        if self._result is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result.save(path)
            messagebox.showinfo("Zapisano", path)


# ══════════════════════════════════════════════════════════════════════════════
# III-C) Mimika twarzy
# ══════════════════════════════════════════════════════════════════════════════

class TabFacialExpression:
    def __init__(self, parent, shared):
        self.shared = shared
        self._result = None
        inner = _scroll(parent)

        tk.Label(inner, text="C) Wykrywanie Mimiki Twarzy",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        tk.Label(
            inner,
            text="Klasyfikuje 7 emocji: szczęście, smutek, złość, zaskoczenie,\n"
                 "strach, wstręt, neutralność.\n"
                 "Backend DeepFace (FER2013) lub MediaPipe FaceLandmarker (Tasks API ≥ 0.10).\n"
                 "Fallback: OpenCV Haar Cascades.",
            fg="#555", justify=tk.CENTER, wraplength=650
        ).pack(pady=4)

        ctrl = _sec(inner, "Parametry")

        tk.Label(ctrl, text="Backend:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.backend = tk.StringVar(value="DeepFace")
        ttk.Combobox(
            ctrl, textvariable=self.backend,
            values=["DeepFace", "MediaPipe + Custom CNN"],
            state="readonly", width=28
        ).grid(row=0, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Rysuj ramki twarzy:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.draw_faces = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.draw_faces).grid(row=1, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Pokaż wynik % emocji:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.show_score = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.show_score).grid(row=2, column=1, sticky=tk.W, padx=8)

        tk.Label(ctrl, text="Punkty twarzy (mesh):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.draw_mesh = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, variable=self.draw_mesh).grid(row=3, column=1, sticky=tk.W, padx=8)

        # info o wymaganiach
        info_frm = tk.LabelFrame(ctrl, text="📦 Wymagane pakiety", padx=8, pady=4)
        info_frm.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=6)
        tk.Label(
            info_frm,
            text="DeepFace:  pip install deepface tf-keras\n"
                 "MediaPipe: pip install mediapipe  (model ~30 MB pobierany automatycznie)",
            fg="#555", font=("Courier", 9), justify=tk.LEFT
        ).pack(anchor=tk.W)

        ttk.Button(ctrl, text="▶  Wykryj emocje",
                   command=self._run).grid(row=5, column=0, columnspan=3, pady=10)

        self.status = tk.Label(inner, text="", fg="#2980b9", font=("Helvetica", 10))
        self.status.pack(pady=2)

        self.dual = DualPreview(inner)
        self.dual.pack(pady=6)

        self.rt = _result_box(inner)
        ttk.Button(inner, text="💾  Zapisz wynik", command=self._save).pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w zakładce A.")
            return
        self.status.config(text="⏳  Analizowanie emocji…", fg="#e67e22")

        params = {
            "backend":     self.backend.get(),
            "draw_faces":  self.draw_faces.get(),
            "show_score":  self.show_score.get(),
            "draw_mesh":   self.draw_mesh.get(),
        }

        def worker():
            try:
                res, info = detect_facial_expression(img, params)
                self._result = res
                self.dual.show(img, res)
                _set_result(self.rt, info)
                self.status.config(text="✅  Zakończono pomyślnie", fg="#27ae60")
            except Exception as e:
                self.status.config(text=f"❌  Błąd: {e}", fg="#e74c3c")
                messagebox.showerror("Błąd", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        if self._result is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self._result.save(path)
            messagebox.showinfo("Zapisano", path)
