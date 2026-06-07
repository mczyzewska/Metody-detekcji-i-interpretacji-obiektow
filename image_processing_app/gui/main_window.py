"""Główne okno aplikacji – nowy wygląd, bez autorów."""
import tkinter as tk
from tkinter import ttk
import os

from gui.tab_basic import (TabLoadSave, TabEdgeDetection, TabThresholding,
                            TabBatchResize, TabWatermark, TabNeuralNetwork)
from gui.tab_advanced import (TabSkeletonDetection, TabTrafficSignDetection,
                               TabFacialExpression)

# ── paleta kolorów ────────────────────────────────────────────────────────────
C_BG        = "#1a1a2e"   # tło główne – granatowy
C_HEADER    = "#16213e"   # pasek górny
C_ACCENT    = "#e94560"   # akcent – różowy/malinowy
C_TAB_BG    = "#0f3460"   # tło paneli
C_TEXT      = "#eaeaea"   # tekst jasny
C_SUBTEXT   = "#a0a0b0"   # tekst drugorzędny
C_STATUS    = "#0d0d1a"   # pasek statusu


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image = None
        self.current_image_path = None
        self._apply_style()
        self._build()

    # ── style ttk ─────────────────────────────────────────────────────────────
    def _apply_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Notebook / zakładki
        style.configure("TNotebook",
                        background=C_BG,
                        borderwidth=0,
                        tabmargins=[2, 4, 0, 0])
        style.configure("TNotebook.Tab",
                        background=C_TAB_BG,
                        foreground=C_TEXT,
                        padding=[12, 6],
                        font=("Helvetica", 9, "bold"),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", C_ACCENT), ("active", "#1a4080")],
                  foreground=[("selected", "white"),  ("active", "white")])

        # Frame
        style.configure("TFrame", background=C_BG)

        # LabelFrame
        style.configure("TLabelframe",
                        background=C_BG,
                        bordercolor=C_ACCENT,
                        relief="flat")
        style.configure("TLabelframe.Label",
                        background=C_BG,
                        foreground=C_ACCENT,
                        font=("Helvetica", 10, "bold"))

        # Button
        style.configure("TButton",
                        background=C_ACCENT,
                        foreground="white",
                        font=("Helvetica", 9, "bold"),
                        padding=[10, 5],
                        relief="flat",
                        borderwidth=0)
        style.map("TButton",
                  background=[("active", "#c73652"), ("pressed", "#a02a40")])

        # Combobox
        style.configure("TCombobox",
                        fieldbackground="#0f3460",
                        background="#0f3460",
                        foreground=C_TEXT,
                        selectbackground=C_ACCENT,
                        arrowcolor=C_ACCENT)

        # Scrollbar
        style.configure("TScrollbar",
                        background=C_TAB_BG,
                        troughcolor=C_BG,
                        arrowcolor=C_ACCENT,
                        borderwidth=0)

        # Progressbar
        style.configure("TProgressbar",
                        background=C_ACCENT,
                        troughcolor="#0f3460",
                        borderwidth=0,
                        thickness=8)

        # Checkbutton
        style.configure("TCheckbutton",
                        background=C_BG,
                        foreground=C_TEXT,
                        font=("Helvetica", 9))
        style.map("TCheckbutton",
                  background=[("active", C_BG)],
                  foreground=[("active", C_TEXT)])

        # Scale
        style.configure("TScale",
                        background=C_BG,
                        troughcolor="#0f3460",
                        sliderlength=16)
        style.map("TScale",
                  background=[("active", C_BG)])

        # Spinbox
        style.configure("TSpinbox",
                        fieldbackground="#0f3460",
                        background="#0f3460",
                        foreground=C_TEXT,
                        arrowcolor=C_ACCENT,
                        borderwidth=0)

        # Entry
        style.configure("TEntry",
                        fieldbackground="#0f3460",
                        foreground=C_TEXT,
                        insertcolor=C_TEXT)

        self.root.configure(bg=C_BG)
        self.root.option_add("*Background", C_BG)
        self.root.option_add("*Foreground", C_TEXT)
        self.root.option_add("*Font", "Helvetica 9")

    # ── budowanie UI ───────────────────────────────────────────────────────────
    def _build(self):
        # ── top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=C_HEADER, height=58)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        # pionowa linia akcentu
        tk.Frame(top, bg=C_ACCENT, width=4).pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(top,
                 text="  🖼  Przetwarzanie Obrazów",
                 bg=C_HEADER, fg="white",
                 font=("Helvetica", 15, "bold")).pack(side=tk.LEFT, padx=12, pady=14)

        tk.Label(top,
                 text="MDiIO 2025/2026",
                 bg=C_HEADER, fg=C_SUBTEXT,
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=16, pady=14)

        # ── separator ────────────────────────────────────────────────────────
        tk.Frame(self.root, bg=C_ACCENT, height=2).pack(fill=tk.X)

        # ── notebook ─────────────────────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        shared = {
            "root":      self.root,
            "get_image": lambda: self.current_image,
            "get_path":  lambda: self.current_image_path,
            "set_image": self._set_image,
        }

        tabs = [
            ("📂  Wczytaj / Zapis",       TabLoadSave),
            ("🔍  Krawędzie",             TabEdgeDetection),
            ("⬛  Progowanie",            TabThresholding),
            ("📐  Rozdzielczość",         TabBatchResize),
            ("💧  Znak wodny",            TabWatermark),
            ("🧠  Sieć neuronowa",        TabNeuralNetwork),
            ("🦴  Szkielet",              TabSkeletonDetection),
            ("🌸  Filtr różowy",           TabTrafficSignDetection),
            ("😊  Mimika twarzy",         TabFacialExpression),
        ]

        for title, Cls in tabs:
            frame = ttk.Frame(nb)
            nb.add(frame, text=title)
            Cls(frame, shared)

        # ── status bar ────────────────────────────────────────────────────────
        self.sv = tk.StringVar(value="Gotowy – wczytaj obraz w zakładce A")
        status = tk.Label(
            self.root,
            textvariable=self.sv,
            bg=C_STATUS, fg=C_SUBTEXT,
            font=("Helvetica", 8),
            anchor=tk.W, padx=10, pady=4,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # linia nad statusem
        tk.Frame(self.root, bg=C_ACCENT, height=1).pack(
            side=tk.BOTTOM, fill=tk.X)

    def _set_image(self, img, path=None):
        self.current_image = img
        if path:
            self.current_image_path = path
        name = os.path.basename(path) if path else "bez nazwy"
        size = f"{img.size[0]}×{img.size[1]} px" if img else ""
        self.sv.set(f"  ●  {name}   |   {size}")
