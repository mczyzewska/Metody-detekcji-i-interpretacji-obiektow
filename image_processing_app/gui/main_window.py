"""
Główne okno aplikacji z zakładkami dla każdej funkcjonalności.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

from gui.tab_basic import (
    TabLoadSave,
    TabEdgeDetection,
    TabThresholding,
    TabResizeBatch,
    TabWatermark,
    TabNeuralNetwork,
)
from gui.tab_advanced import (
    TabSkeletonDetection,
    TabTrafficSignDetection,
    TabFacialExpression,
)


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image = None        # PIL Image
        self.current_image_path = None   # str path
        self.tk_image = None             # PhotoImage for display

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#2c3e50", height=50)
        top.pack(fill=tk.X)
        tk.Label(
            top,
            text="🖼  Aplikacja do Przetwarzania Obrazów",
            bg="#2c3e50",
            fg="white",
            font=("Helvetica", 14, "bold"),
            pady=10,
        ).pack(side=tk.LEFT, padx=15)

        # authors label
        tk.Label(
            top,
            text="M. Czyżewska 21227 | A. Witów 21319",
            bg="#2c3e50",
            fg="#bdc3c7",
            font=("Helvetica", 9),
            pady=10,
        ).pack(side=tk.RIGHT, padx=15)

        # ── notebook ─────────────────────────────────────────────────
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # shared state passed to every tab
        shared = {
            "root": self.root,
            "get_image": lambda: self.current_image,
            "get_path": lambda: self.current_image_path,
            "set_image": self._set_image,
        }

        tabs = [
            ("📂 Wczytaj/Zapis",      TabLoadSave),
            ("🔍 Krawędzie",          TabEdgeDetection),
            ("⬛ Progowanie",         TabThresholding),
            ("📐 Zmiana rozdzielczości", TabResizeBatch),
            ("💧 Znak wodny",         TabWatermark),
            ("🧠 Sieć neuronowa",     TabNeuralNetwork),
            ("🦴 Szkielet człowieka", TabSkeletonDetection),
            ("🚦 Znaki drogowe",      TabTrafficSignDetection),
            ("😊 Mimika twarzy",      TabFacialExpression),
        ]

        for title, TabClass in tabs:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            TabClass(frame, shared)

        # ── status bar ───────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Gotowy")
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#ecf0f1",
            fg="#2c3e50",
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    def _set_image(self, img, path=None):
        self.current_image = img
        if path:
            self.current_image_path = path
        self.status_var.set(
            f"Załadowany obraz: {os.path.basename(path) if path else 'brak nazwy'}"
            + (f"  |  Rozmiar: {img.size[0]}×{img.size[1]}" if img else "")
        )
