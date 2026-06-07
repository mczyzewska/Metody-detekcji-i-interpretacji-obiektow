"""
Zakładki zaawansowane – ciemny motyw:
  III-A) Szkielet człowieka
  III-B) Filtr różowy (dawniej znaki drogowe)
  III-C) Mimika twarzy
"""
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from gui.image_preview import DualPreview
from processing.skeleton_detection import detect_skeleton
from processing.traffic_signs import detect_traffic_signs, PINK_MODES
from processing.facial_expression import detect_facial_expression

C_BG    = "#1a1a2e"
C_PANEL = "#16213e"
C_INPUT = "#0f3460"
C_ACC   = "#e94560"
C_TEXT  = "#eaeaea"
C_SUB   = "#a0a0b0"
C_OK    = "#2ecc71"
C_ERR   = "#e74c3c"
C_WARN  = "#f39c12"


def _scroll(parent):
    outer = tk.Frame(parent, bg=C_BG); outer.pack(fill=tk.BOTH, expand=True)
    c  = tk.Canvas(outer, bg=C_BG, borderwidth=0, highlightthickness=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=c.yview)
    c.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y); c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(c, bg=C_BG)
    wid = c.create_window((0, 0), window=inner, anchor="nw")
    def _cfg(e):
        c.configure(scrollregion=c.bbox("all"))
        c.itemconfig(wid, width=c.winfo_width())
    inner.bind("<Configure>", _cfg); c.bind("<Configure>", _cfg)
    return inner


def _title(parent, text):
    tk.Label(parent, text=text, bg=C_BG, fg="white",
             font=("Helvetica", 15, "bold")).pack(pady=(16, 4))
    tk.Frame(parent, bg=C_ACC, height=2).pack(fill=tk.X, padx=20, pady=(0, 10))


def _sec(parent, text):
    f = tk.LabelFrame(parent, text=f"  {text}  ", bg=C_PANEL, fg=C_ACC,
                      font=("Helvetica", 9, "bold"), bd=0, relief="flat",
                      highlightbackground=C_INPUT, highlightthickness=1,
                      padx=12, pady=10)
    f.pack(padx=16, pady=6, fill=tk.X)
    return f


def _label(parent, text, row):
    tk.Label(parent, text=text, bg=C_PANEL, fg=C_SUB,
             font=("Helvetica", 9)).grid(row=row, column=0, sticky=tk.W, pady=3, padx=(0, 8))


def _check(parent, var, row):
    tk.Checkbutton(parent, variable=var, bg=C_PANEL, fg=C_TEXT,
                   selectcolor=C_INPUT, activebackground=C_PANEL,
                   activeforeground=C_TEXT).grid(row=row, column=1, sticky=tk.W, padx=(0, 6))


def _run_btn(parent, text, cmd):
    tk.Button(parent, text=text, command=cmd,
              bg=C_ACC, fg="white", activebackground="#c73652",
              font=("Helvetica", 9, "bold"), relief="flat", bd=0,
              padx=14, pady=6, cursor="hand2").grid(
        row=99, column=0, columnspan=3, pady=10)


def _result_box(parent):
    rf = tk.LabelFrame(parent, text="  Wyniki  ", bg=C_PANEL, fg=C_ACC,
                       font=("Helvetica", 9, "bold"), bd=0, relief="flat",
                       highlightbackground=C_INPUT, highlightthickness=1,
                       padx=8, pady=6)
    rf.pack(fill=tk.X, padx=16, pady=4)
    sb = ttk.Scrollbar(rf); sb.pack(side=tk.RIGHT, fill=tk.Y)
    t  = tk.Text(rf, height=7, state=tk.DISABLED,
                 bg=C_PANEL, fg=C_SUB, relief="flat", bd=0,
                 font=("Courier", 9), yscrollcommand=sb.set)
    t.pack(fill=tk.BOTH); sb.config(command=t.yview)
    return t


def _set_result(tb, text):
    tb.config(state=tk.NORMAL); tb.delete("1.0", tk.END)
    tb.insert(tk.END, text); tb.config(state=tk.DISABLED)


def _save_btn(parent, get_img):
    def _save():
        img = get_img()
        if img is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj operację."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: img.save(path); messagebox.showinfo("Zapisano", path)
    tk.Button(parent, text="💾  Zapisz wynik", command=_save,
              bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
              relief="flat", bd=0, padx=12, pady=5, cursor="hand2").pack(pady=6)


# ══════════════════════════════════════════════════════════════════════════════
# III-A Szkielet człowieka
# ══════════════════════════════════════════════════════════════════════════════
class TabSkeletonDetection:
    def __init__(self, parent, shared):
        self.shared = shared; self._result = None
        inner = _scroll(parent)
        _title(inner, "Wykrywanie Szkieletu Człowieka")
        tk.Label(inner,
                 text="MediaPipe PoseLandmarker (Tasks API ≥ 0.10) – 33 punkty kluczowe ciała.\n"
                      "Model (~30 MB) pobierany automatycznie przy pierwszym użyciu.",
                 bg=C_BG, fg=C_SUB, font=("Helvetica", 9),
                 justify=tk.CENTER, wraplength=660).pack(pady=4)

        ctrl = _sec(inner, "Parametry")
        _label(ctrl, "Min. pewność detekcji:", 0)
        self.det = tk.DoubleVar(value=0.5)
        frm0 = tk.Frame(ctrl, bg=C_PANEL); frm0.grid(row=0, column=1, sticky=tk.W, padx=(0,6), pady=3)
        tk.Scale(frm0, from_=0.1, to=1.0, variable=self.det, orient=tk.HORIZONTAL, length=200,
                 resolution=0.05,
                 bg=C_PANEL, fg=C_TEXT, troughcolor=C_INPUT, activebackground=C_ACC,
                 highlightthickness=0, bd=0).pack(side=tk.LEFT)
        tk.Label(frm0, textvariable=self.det, width=4, bg=C_PANEL, fg=C_TEXT).pack(side=tk.LEFT)

        _label(ctrl, "Min. pewność śledzenia:", 1)
        self.trk = tk.DoubleVar(value=0.5)
        frm1 = tk.Frame(ctrl, bg=C_PANEL); frm1.grid(row=1, column=1, sticky=tk.W, padx=(0,6), pady=3)
        tk.Scale(frm1, from_=0.1, to=1.0, variable=self.trk, orient=tk.HORIZONTAL, length=200,
                 resolution=0.05,
                 bg=C_PANEL, fg=C_TEXT, troughcolor=C_INPUT, activebackground=C_ACC,
                 highlightthickness=0, bd=0).pack(side=tk.LEFT)
        tk.Label(frm1, textvariable=self.trk, width=4, bg=C_PANEL, fg=C_TEXT).pack(side=tk.LEFT)

        _label(ctrl, "Rysuj punkty kluczowe:", 2)
        self.draw_lm = tk.BooleanVar(value=True); _check(ctrl, self.draw_lm, 2)
        _label(ctrl, "Rysuj połączenia:", 3)
        self.draw_cn = tk.BooleanVar(value=True); _check(ctrl, self.draw_cn, 3)
        _run_btn(ctrl, "▶  Wykryj szkielet", self._run)

        self.st = tk.Label(inner, text="", bg=C_BG, fg=C_SUB, font=("Helvetica", 9)); self.st.pack(pady=2)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        self.rt = _result_box(inner)
        _save_btn(inner, lambda: self._result)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        self.st.config(text="⏳  Wykrywanie szkieletu…", fg=C_WARN)
        params = {"det_confidence": round(self.det.get(),2),
                  "track_confidence": round(self.trk.get(),2),
                  "draw_landmarks": self.draw_lm.get(),
                  "draw_connections": self.draw_cn.get()}
        def worker():
            try:
                res, info = detect_skeleton(img, params)
                self._result = res; self.dual.show(img, res)
                _set_result(self.rt, info); self.st.config(text="✅  Zakończono", fg=C_OK)
            except Exception as e:
                self.st.config(text=f"❌  {e}", fg=C_ERR); messagebox.showerror("Błąd", str(e))
        threading.Thread(target=worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# III-B Filtr różowy
# ══════════════════════════════════════════════════════════════════════════════
class TabTrafficSignDetection:
    def __init__(self, parent, shared):
        self.shared = shared; self._result = None
        inner = _scroll(parent)
        _title(inner, "Filtr Różowy 🌸")
        tk.Label(inner,
                 text="Konwertuje obraz na odcienie różowego.\n"
                      "Wybierz tryb i intensywność efektu.",
                 bg=C_BG, fg=C_SUB, font=("Helvetica", 9),
                 justify=tk.CENTER, wraplength=660).pack(pady=4)

        ctrl = _sec(inner, "Parametry")

        _label(ctrl, "Tryb różowego:", 0)
        self.mode = tk.StringVar(value="Klasyczny różowy")
        cb = ttk.Combobox(ctrl, textvariable=self.mode, values=PINK_MODES,
                          state="readonly", width=22)
        cb.grid(row=0, column=1, sticky=tk.W, padx=(0,6), pady=3)

        _label(ctrl, "Intensywność:", 1)
        self.intensity = tk.DoubleVar(value=0.8)
        frm = tk.Frame(ctrl, bg=C_PANEL); frm.grid(row=1, column=1, sticky=tk.W, padx=(0,6), pady=3)
        tk.Scale(frm, from_=0.05, to=1.0, variable=self.intensity,
                 orient=tk.HORIZONTAL, length=220, resolution=0.05,
                 bg=C_PANEL, fg=C_TEXT, troughcolor=C_INPUT, activebackground=C_ACC,
                 highlightthickness=0, bd=0).pack(side=tk.LEFT)
        tk.Label(frm, textvariable=self.intensity, width=4, bg=C_PANEL, fg=C_TEXT).pack(side=tk.LEFT)

        _label(ctrl, "Rozmycie (Gaussian):", 2)
        self.blur = tk.BooleanVar(value=False); _check(ctrl, self.blur, 2)

        _label(ctrl, "Winietowanie:", 3)
        self.vignette = tk.BooleanVar(value=False); _check(ctrl, self.vignette, 3)

        _run_btn(ctrl, "▶  Zastosuj filtr różowy", self._run)

        self.st = tk.Label(inner, text="", bg=C_BG, fg=C_SUB, font=("Helvetica", 9)); self.st.pack(pady=2)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        self.rt = _result_box(inner)
        _save_btn(inner, lambda: self._result)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        self.st.config(text="⏳  Nakładanie filtra…", fg=C_WARN)
        params = {"pink_mode":  self.mode.get(),
                  "intensity":  round(self.intensity.get(), 2),
                  "blur":       self.blur.get(),
                  "vignette":   self.vignette.get()}
        def worker():
            try:
                res, info = detect_traffic_signs(img, params)
                self._result = res; self.dual.show(img, res)
                _set_result(self.rt, info); self.st.config(text="✅  Gotowe", fg=C_OK)
            except Exception as e:
                self.st.config(text=f"❌  {e}", fg=C_ERR); messagebox.showerror("Błąd", str(e))
        threading.Thread(target=worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# III-C Mimika twarzy
# ══════════════════════════════════════════════════════════════════════════════
class TabFacialExpression:
    def __init__(self, parent, shared):
        self.shared = shared; self._result = None
        inner = _scroll(parent)
        _title(inner, "Wykrywanie Mimiki Twarzy")
        tk.Label(inner,
                 text="Klasyfikuje 7 emocji: szczęście, smutek, złość, zaskoczenie,\n"
                      "strach, wstręt, neutralność.\n"
                      "Backend: DeepFace (FER2013) lub MediaPipe FaceLandmarker (Tasks API ≥ 0.10).",
                 bg=C_BG, fg=C_SUB, font=("Helvetica", 9),
                 justify=tk.CENTER, wraplength=660).pack(pady=4)

        ctrl = _sec(inner, "Parametry")
        _label(ctrl, "Backend:", 0)
        self.backend = tk.StringVar(value="DeepFace")
        ttk.Combobox(ctrl, textvariable=self.backend,
                     values=["DeepFace", "MediaPipe + Custom CNN"],
                     state="readonly", width=26).grid(row=0, column=1, sticky=tk.W, padx=(0,6), pady=3)

        _label(ctrl, "Rysuj ramki twarzy:", 1)
        self.draw_faces = tk.BooleanVar(value=True); _check(ctrl, self.draw_faces, 1)
        _label(ctrl, "Pokaż wynik % emocji:", 2)
        self.show_score = tk.BooleanVar(value=True); _check(ctrl, self.show_score, 2)
        _label(ctrl, "Punkty twarzy (mesh):", 3)
        self.draw_mesh = tk.BooleanVar(value=False); _check(ctrl, self.draw_mesh, 3)

        # info box
        info_frm = tk.LabelFrame(ctrl, text="  Wymagane pakiety  ", bg=C_PANEL, fg=C_ACC,
                                  font=("Helvetica", 8, "bold"), bd=0, relief="flat",
                                  highlightbackground=C_INPUT, highlightthickness=1,
                                  padx=8, pady=4)
        info_frm.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=6)
        tk.Label(info_frm,
                 text="DeepFace:  pip install deepface tf-keras\n"
                      "MediaPipe: pip install mediapipe",
                 bg=C_PANEL, fg=C_SUB, font=("Courier", 8), justify=tk.LEFT).pack(anchor=tk.W)

        _run_btn(ctrl, "▶  Wykryj emocje", self._run)

        self.st = tk.Label(inner, text="", bg=C_BG, fg=C_SUB, font=("Helvetica", 9)); self.st.pack(pady=2)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        self.rt = _result_box(inner)
        _save_btn(inner, lambda: self._result)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        self.st.config(text="⏳  Analizowanie emocji…", fg=C_WARN)
        params = {"backend": self.backend.get(), "draw_faces": self.draw_faces.get(),
                  "show_score": self.show_score.get(), "draw_mesh": self.draw_mesh.get()}
        def worker():
            try:
                res, info = detect_facial_expression(img, params)
                self._result = res; self.dual.show(img, res)
                _set_result(self.rt, info); self.st.config(text="✅  Zakończono", fg=C_OK)
            except Exception as e:
                self.st.config(text=f"❌  {e}", fg=C_ERR); messagebox.showerror("Błąd", str(e))
        threading.Thread(target=worker, daemon=True).start()
