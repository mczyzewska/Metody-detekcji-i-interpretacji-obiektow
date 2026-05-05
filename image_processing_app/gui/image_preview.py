"""
Reusable image preview panel used across tabs.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ImagePreview(tk.Frame):
    """Shows a before/after image side by side (or single image)."""

    def __init__(self, parent, label="Podgląd", max_size=(500, 400)):
        super().__init__(parent, bg="#ecf0f1")
        self.max_size = max_size
        self._tk_img = None

        tk.Label(self, text=label, font=("Helvetica", 10, "bold"), bg="#ecf0f1").pack()
        self.canvas = tk.Label(
            self, bg="#bdc3c7", width=max_size[0], height=max_size[1], text="Brak obrazu"
        )
        self.canvas.pack(padx=4, pady=4)

    def show(self, pil_image: Image.Image | None):
        if pil_image is None:
            self.canvas.config(image="", text="Brak obrazu")
            self._tk_img = None
            return

        img = pil_image.copy()
        img.thumbnail(self.max_size, Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._tk_img, text="")


class DualPreview(tk.Frame):
    """Two side-by-side previews: original | wynik."""

    def __init__(self, parent):
        super().__init__(parent, bg="#ecf0f1")
        self.left = ImagePreview(self, label="Oryginał", max_size=(380, 320))
        self.left.pack(side=tk.LEFT, padx=5, pady=5)
        self.right = ImagePreview(self, label="Wynik", max_size=(380, 320))
        self.right.pack(side=tk.LEFT, padx=5, pady=5)

    def show(self, original, result):
        self.left.show(original)
        self.right.show(result)
