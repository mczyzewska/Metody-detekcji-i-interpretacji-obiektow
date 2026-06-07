"""Zakładki A–F – ciemny motyw."""
import os, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gui.image_preview import DualPreview, ImagePreview
from processing.edge_detection import detect_edges
from processing.thresholding import threshold_image
from processing.batch_resize import batch_resize
from processing.watermark import batch_watermark
from processing.neural_network import run_neural_network

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
    outer = tk.Frame(parent, bg=C_BG)
    outer.pack(fill=tk.BOTH, expand=True)
    c  = tk.Canvas(outer, bg=C_BG, borderwidth=0, highlightthickness=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=c.yview)
    c.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(c, bg=C_BG)
    wid   = c.create_window((0, 0), window=inner, anchor="nw")
    def _cfg(e):
        c.configure(scrollregion=c.bbox("all"))
        c.itemconfig(wid, width=c.winfo_width())
    inner.bind("<Configure>", _cfg)
    c.bind("<Configure>", _cfg)
    return inner

def _title(parent, text):
    tk.Label(parent, text=text, bg=C_BG, fg="white",
             font=("Helvetica", 15, "bold")).pack(pady=(16, 4))
    tk.Frame(parent, bg=C_ACC, height=2).pack(fill=tk.X, padx=20, pady=(0, 10))

def _sec(parent, text):
    f = tk.LabelFrame(parent, text=f"  {text}  ", bg=C_PANEL, fg=C_ACC,
                      font=("Helvetica", 9, "bold"),
                      bd=0, relief="flat",
                      highlightbackground=C_INPUT, highlightthickness=1,
                      padx=12, pady=10)
    f.pack(padx=16, pady=6, fill=tk.X)
    return f

def _label(parent, text, row, col=0, colspan=1):
    tk.Label(parent, text=text, bg=C_PANEL, fg=C_SUB,
             font=("Helvetica", 9)).grid(row=row, column=col, sticky=tk.W, pady=3,
                                         padx=(0, 8), columnspan=colspan)

def _entry(parent, var, row, width=32):
    e = tk.Entry(parent, textvariable=var, width=width,
                 bg=C_INPUT, fg=C_TEXT, insertbackground=C_TEXT,
                 relief="flat", bd=4)
    e.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=3)
    return e

def _spinbox(parent, var, row, from_=0, to=500, width=9, increment=1):
    s = tk.Spinbox(parent, from_=from_, to=to, increment=increment,
                   textvariable=var, width=width,
                   bg=C_INPUT, fg=C_TEXT, buttonbackground=C_INPUT,
                   relief="flat", bd=2)
    s.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=3)
    return s

def _combo(parent, var, values, row, width=20):
    c = ttk.Combobox(parent, textvariable=var, values=values,
                     state="readonly", width=width)
    c.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=3)
    return c

def _check(parent, var, row):
    cb = tk.Checkbutton(parent, variable=var, bg=C_PANEL,
                        fg=C_TEXT, selectcolor=C_INPUT,
                        activebackground=C_PANEL, activeforeground=C_TEXT)
    cb.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=3)

def _btn(parent, text, cmd, full=False):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  activeforeground="white", font=("Helvetica", 9, "bold"),
                  relief="flat", bd=0, padx=14, pady=6, cursor="hand2")
    if full:
        b.pack(fill=tk.X, padx=16, pady=6)
    return b

def _status_lbl(parent):
    l = tk.Label(parent, text="", bg=C_BG, fg=C_SUB,
                 font=("Helvetica", 9))
    l.pack(pady=2)
    return l

def _logbox(parent):
    f = tk.Frame(parent, bg=C_BG)
    f.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)
    sb = ttk.Scrollbar(f)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    t = tk.Text(f, height=8, state=tk.DISABLED,
                bg=C_PANEL, fg=C_SUB, insertbackground=C_TEXT,
                relief="flat", bd=0, font=("Courier", 9),
                yscrollcommand=sb.set)
    t.pack(fill=tk.BOTH, expand=True)
    sb.config(command=t.yview)
    return t

def _log(tb, msg):
    tb.config(state=tk.NORMAL)
    tb.insert(tk.END, msg + "\n")
    tb.see(tk.END)
    tb.config(state=tk.DISABLED)

def _folder_row(ctrl, lbl, var, row):
    _label(ctrl, lbl, row)
    tk.Entry(ctrl, textvariable=var, width=36,
             bg=C_INPUT, fg=C_TEXT, insertbackground=C_TEXT,
             relief="flat", bd=4).grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=3)
    tk.Button(ctrl, text="📁", command=lambda v=var: v.set(filedialog.askdirectory() or v.get()),
              bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
              relief="flat", bd=0, padx=6, cursor="hand2").grid(row=row, column=2)


# ══════════════════════════════════════════════════════════════════════════════
class TabLoadSave:
    def __init__(self, parent, shared):
        self.shared = shared
        inner = _scroll(parent)
        _title(inner, "Wczytywanie i Zapis Obrazów")

        btn_row = tk.Frame(inner, bg=C_BG)
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="📂  Wczytaj obraz", command=self._load,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 10, "bold"), relief="flat", bd=0,
                  padx=20, pady=8, cursor="hand2").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_row, text="💾  Zapisz obraz", command=self._save,
                  bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
                  font=("Helvetica", 10, "bold"), relief="flat", bd=0,
                  padx=20, pady=8, cursor="hand2").pack(side=tk.LEFT, padx=10)

        self.info = tk.Label(inner, text="Brak załadowanego obrazu",
                             bg=C_BG, fg=C_SUB, font=("Helvetica", 10))
        self.info.pack(pady=4)
        self.prev = ImagePreview(inner, label="Podgląd", max_size=(700, 500))
        self.prev.pack(pady=10)

    def _load(self):
        path = filedialog.askopenfilename(title="Wybierz obraz",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif *.webp"),
                       ("Wszystkie", "*.*")])
        if not path: return
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            self.shared["set_image"](img, path)
            self.info.config(
                text=f"✅  {os.path.basename(path)}   |   {img.size[0]}×{img.size[1]} px",
                fg=C_OK)
            self.prev.show(img)
        except Exception as e:
            messagebox.showerror("Błąd wczytywania", str(e))

    def _save(self):
        img = self.shared["get_image"]()
        if img is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp"),("TIFF","*.tiff")])
        if path:
            try: img.save(path); messagebox.showinfo("Zapisano", path)
            except Exception as e: messagebox.showerror("Błąd zapisu", str(e))


# ══════════════════════════════════════════════════════════════════════════════
class TabEdgeDetection:
    ALGOS = ["Canny", "Sobel", "Laplacian"]
    def __init__(self, parent, shared):
        self.shared = shared; self._res = None
        inner = _scroll(parent)
        _title(inner, "Wykrywanie Krawędzi")
        ctrl = _sec(inner, "Parametry")
        _label(ctrl, "Algorytm:", 0)
        self.algo = tk.StringVar(value="Canny")
        _combo(ctrl, self.algo, self.ALGOS, 0, width=16)
        _label(ctrl, "Próg dolny (Canny):", 1)
        self.low = tk.IntVar(value=50)
        _spinbox(ctrl, self.low, 1, 0, 500)
        _label(ctrl, "Próg górny (Canny):", 2)
        self.high = tk.IntVar(value=150)
        _spinbox(ctrl, self.high, 2, 0, 500)
        tk.Button(ctrl, text="▶  Wykryj krawędzie", command=self._run,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 9, "bold"), relief="flat", bd=0,
                  padx=14, pady=6, cursor="hand2").grid(row=3, column=0, columnspan=3, pady=10)
        self.st = _status_lbl(inner)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        tk.Button(inner, text="💾  Zapisz wynik", command=self._save,
                  bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
                  relief="flat", bd=0, padx=12, pady=5, cursor="hand2").pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        try:
            r = detect_edges(img, self.algo.get(), {"low": self.low.get(), "high": self.high.get()})
            self._res = r; self.dual.show(img, r)
            self.st.config(text=f"✅  {self.algo.get()} – zakończono", fg=C_OK)
        except Exception as e: messagebox.showerror("Błąd", str(e))

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku", "Najpierw wykonaj detekcję."); return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano", path)


# ══════════════════════════════════════════════════════════════════════════════
class TabThresholding:
    ALGOS = ["Globalne (Otsu)", "Adaptacyjne (Mean)", "Adaptacyjne (Gaussian)"]
    def __init__(self, parent, shared):
        self.shared = shared; self._res = None
        inner = _scroll(parent)
        _title(inner, "Progowanie Obrazu")
        ctrl = _sec(inner, "Parametry")
        _label(ctrl, "Algorytm:", 0)
        self.algo = tk.StringVar(value="Globalne (Otsu)")
        _combo(ctrl, self.algo, self.ALGOS, 0, width=26)
        _label(ctrl, "Rozmiar bloku (adapt.):", 1)
        self.bs = tk.IntVar(value=11)
        _spinbox(ctrl, self.bs, 1, 3, 99, increment=2)
        _label(ctrl, "Stała C (adapt.):", 2)
        self.C = tk.IntVar(value=2)
        _spinbox(ctrl, self.C, 2, 0, 50)
        tk.Button(ctrl, text="▶  Proguj", command=self._run,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 9, "bold"), relief="flat", bd=0,
                  padx=14, pady=6, cursor="hand2").grid(row=3, column=0, columnspan=3, pady=10)
        self.st = _status_lbl(inner)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        tk.Button(inner, text="💾  Zapisz wynik", command=self._save,
                  bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
                  relief="flat", bd=0, padx=12, pady=5, cursor="hand2").pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        try:
            r = threshold_image(img, self.algo.get(), {"block_size": self.bs.get(), "C": self.C.get()})
            self._res = r; self.dual.show(img, r)
            self.st.config(text=f"✅  {self.algo.get()}", fg=C_OK)
        except Exception as e: messagebox.showerror("Błąd", str(e))

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku", "Najpierw wykonaj progowanie."); return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano", path)


# ══════════════════════════════════════════════════════════════════════════════
class TabBatchResize:
    METHODS = ["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"]
    def __init__(self, parent, shared):
        self.shared = shared
        inner = _scroll(parent)
        _title(inner, "Zmiana Rozdzielczości – Wsadowo")
        ctrl = _sec(inner, "Ustawienia")
        self.src = tk.StringVar(); self.dst = tk.StringVar()
        _folder_row(ctrl, "Folder wejściowy:", self.src, 0)
        _folder_row(ctrl, "Folder wyjściowy:", self.dst, 1)
        _label(ctrl, "Szerokość (px):", 2)
        self.w = tk.IntVar(value=800); _spinbox(ctrl, self.w, 2, 1, 9999)
        _label(ctrl, "Wysokość (px):", 3)
        self.h = tk.IntVar(value=600); _spinbox(ctrl, self.h, 3, 1, 9999)
        _label(ctrl, "Zachowaj proporcje:", 4)
        self.rat = tk.BooleanVar(value=True); _check(ctrl, self.rat, 4)
        _label(ctrl, "Interpolacja:", 5)
        self.mth = tk.StringVar(value="LANCZOS"); _combo(ctrl, self.mth, self.METHODS, 5, 14)
        tk.Button(ctrl, text="▶  Zmień rozdzielczość", command=self._run,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 9, "bold"), relief="flat", bd=0,
                  padx=14, pady=6, cursor="hand2").grid(row=6, column=0, columnspan=3, pady=10)
        self.prog = ttk.Progressbar(inner, mode="determinate")
        self.prog.pack(fill=tk.X, padx=16, pady=4)
        self.log = _logbox(inner)

    def _run(self):
        if not self.src.get() or not self.dst.get():
            messagebox.showwarning("Brak folderów", "Wybierz foldery."); return
        self.log.config(state=tk.NORMAL); self.log.delete("1.0", tk.END); self.log.config(state=tk.DISABLED)
        self.prog["value"] = 0
        params = {"width": self.w.get(), "height": self.h.get(),
                  "keep_ratio": self.rat.get(), "method": self.mth.get()}
        def worker():
            try:
                batch_resize(self.src.get(), self.dst.get(), params,
                             progress_cb=lambda p: self.prog.config(value=p),
                             log_cb=lambda m: _log(self.log, m))
                _log(self.log, "✅ Zakończono."); messagebox.showinfo("Gotowe", "Zmiana rozdzielczości zakończona.")
            except Exception as e: messagebox.showerror("Błąd", str(e))
        threading.Thread(target=worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
class TabWatermark:
    POSITIONS = ["Środek","Lewy-górny","Prawy-górny","Lewy-dolny","Prawy-dolny"]
    def __init__(self, parent, shared):
        self.shared = shared
        inner = _scroll(parent)
        _title(inner, "Dodanie Znaku Wodnego – Wsadowo")
        ctrl = _sec(inner, "Ustawienia")
        self.src = tk.StringVar(); self.dst = tk.StringVar()
        _folder_row(ctrl, "Folder wejściowy:", self.src, 0)
        _folder_row(ctrl, "Folder wyjściowy:", self.dst, 1)
        _label(ctrl, "Tekst:", 2)
        self.txt = tk.StringVar(value="© 2026"); _entry(ctrl, self.txt, 2, 26)
        _label(ctrl, "Pozycja:", 3)
        self.pos = tk.StringVar(value="Prawy-dolny"); _combo(ctrl, self.pos, self.POSITIONS, 3, 14)
        _label(ctrl, "Przezroczystość (0–255):", 4)
        self.alp = tk.IntVar(value=140); _spinbox(ctrl, self.alp, 4, 0, 255)
        _label(ctrl, "Rozmiar czcionki:", 5)
        self.fs = tk.IntVar(value=36); _spinbox(ctrl, self.fs, 5, 8, 200)
        _label(ctrl, "Kolor (#RRGGBB):", 6)
        self.col = tk.StringVar(value="#FFFFFF"); _entry(ctrl, self.col, 6, 10)
        tk.Button(ctrl, text="▶  Dodaj znaki wodne", command=self._run,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 9, "bold"), relief="flat", bd=0,
                  padx=14, pady=6, cursor="hand2").grid(row=7, column=0, columnspan=3, pady=10)
        self.prog = ttk.Progressbar(inner, mode="determinate")
        self.prog.pack(fill=tk.X, padx=16, pady=4)
        self.log = _logbox(inner)

    def _run(self):
        if not self.src.get() or not self.dst.get():
            messagebox.showwarning("Brak folderów", "Wybierz foldery."); return
        self.log.config(state=tk.NORMAL); self.log.delete("1.0", tk.END); self.log.config(state=tk.DISABLED)
        params = {"text":self.txt.get(),"position":self.pos.get(),"alpha":self.alp.get(),
                  "font_size":self.fs.get(),"color":self.col.get()}
        def worker():
            try:
                batch_watermark(self.src.get(), self.dst.get(), params,
                                progress_cb=lambda p: self.prog.config(value=p),
                                log_cb=lambda m: _log(self.log, m))
                _log(self.log, "✅ Zakończono."); messagebox.showinfo("Gotowe", "Znaki wodne dodane.")
            except Exception as e: messagebox.showerror("Błąd", str(e))
        threading.Thread(target=worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
class TabNeuralNetwork:
    MODELS = ["Klasyfikacja (MobileNetV2 – ImageNet)",
              "Segmentacja semantyczna (DeepLab v3)",
              "Detekcja obiektów (YOLOv5 / Haar fallback)"]
    def __init__(self, parent, shared):
        self.shared = shared; self._res = None
        inner = _scroll(parent)
        _title(inner, "Przetwarzanie – Sztuczna Sieć Neuronowa")
        ctrl = _sec(inner, "Parametry")
        _label(ctrl, "Model:", 0)
        self.model = tk.StringVar(value=self.MODELS[0])
        _combo(ctrl, self.model, self.MODELS, 0, 44)
        _label(ctrl, "Próg pewności:", 1)
        self.conf = tk.DoubleVar(value=0.5)
        frm = tk.Frame(ctrl, bg=C_PANEL); frm.grid(row=1, column=1, sticky=tk.W, padx=(0, 6), pady=3)
        tk.Scale(frm, from_=0.05, to=0.99, variable=self.conf, orient=tk.HORIZONTAL, length=220,
                 resolution=0.05,
                 bg=C_PANEL, fg=C_TEXT, troughcolor=C_INPUT, activebackground=C_ACC,
                 highlightthickness=0, bd=0).pack(side=tk.LEFT)
        tk.Label(frm, textvariable=self.conf, width=5, bg=C_PANEL, fg=C_TEXT).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="▶  Uruchom model", command=self._run,
                  bg=C_ACC, fg="white", activebackground="#c73652",
                  font=("Helvetica", 9, "bold"), relief="flat", bd=0,
                  padx=14, pady=6, cursor="hand2").grid(row=2, column=0, columnspan=3, pady=10)
        self.st = _status_lbl(inner)
        self.dual = DualPreview(inner); self.dual.pack(pady=4)
        rf = tk.LabelFrame(inner, text="  Wyniki modelu  ", bg=C_PANEL, fg=C_ACC,
                           font=("Helvetica", 9, "bold"), bd=0, relief="flat",
                           highlightbackground=C_INPUT, highlightthickness=1, padx=8, pady=6)
        rf.pack(fill=tk.X, padx=16, pady=4)
        sb2 = ttk.Scrollbar(rf); sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.rt = tk.Text(rf, height=6, state=tk.DISABLED,
                          bg=C_PANEL, fg=C_SUB, relief="flat", bd=0,
                          font=("Courier", 9), yscrollcommand=sb2.set)
        self.rt.pack(fill=tk.BOTH); sb2.config(command=self.rt.yview)
        tk.Button(inner, text="💾  Zapisz wynik", command=self._save,
                  bg=C_INPUT, fg=C_TEXT, activebackground=C_ACC,
                  relief="flat", bd=0, padx=12, pady=5, cursor="hand2").pack(pady=6)

    def _run(self):
        img = self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz."); return
        self.st.config(text="⏳  Przetwarzanie…", fg=C_WARN)
        def worker():
            try:
                r, info = run_neural_network(img, self.model.get(), round(self.conf.get(), 2))
                self._res = r; self.dual.show(img, r)
                self.rt.config(state=tk.NORMAL); self.rt.delete("1.0", tk.END)
                self.rt.insert(tk.END, info); self.rt.config(state=tk.DISABLED)
                self.st.config(text="✅  Zakończono", fg=C_OK)
            except Exception as e:
                self.st.config(text=f"❌  {e}", fg=C_ERR)
                messagebox.showerror("Błąd modelu", str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku", "Najpierw uruchom model."); return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano", path)
