"""Widgety podglądu obrazów."""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ImagePreview(tk.LabelFrame):
    def __init__(self, parent, label="Podgląd", max_size=(420, 340)):
        super().__init__(parent, text=label, padx=4, pady=4)
        self.max_size = max_size
        self._tk = None

        # Canvas o stałym rozmiarze w pikselach – obraz zawsze wypełnia widget
        self._canvas = tk.Canvas(
            self,
            width=max_size[0],
            height=max_size[1],
            bg="#bdc3c7",
            highlightthickness=0,
        )
        self._canvas.pack()

        # Napis "Brak obrazu" wyśrodkowany na pustym canvasie
        self._placeholder = self._canvas.create_text(
            max_size[0] // 2,
            max_size[1] // 2,
            text="Brak obrazu",
            fill="#555555",
            font=("TkDefaultFont", 11),
        )
        self._image_item = None

    def show(self, pil_image):
        if pil_image is None:
            # Usuń poprzedni obraz, pokaż placeholder
            if self._image_item is not None:
                self._canvas.delete(self._image_item)
                self._image_item = None
            self._canvas.itemconfigure(self._placeholder, state="normal")
            self._tk = None
            return

        # Skaluj z zachowaniem proporcji, żeby mieścił się w max_size
        img = pil_image.copy()
        img.thumbnail(self.max_size, Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(img)

        # Wyśrodkuj obraz na canvasie
        cx = self.max_size[0] // 2
        cy = self.max_size[1] // 2

        if self._image_item is None:
            self._image_item = self._canvas.create_image(cx, cy, anchor="center", image=self._tk)
        else:
            self._canvas.itemconfigure(self._image_item, image=self._tk)
            self._canvas.coords(self._image_item, cx, cy)

        # Ukryj placeholder gdy obraz jest widoczny
        self._canvas.itemconfigure(self._placeholder, state="hidden")


class DualPreview(tk.Frame):
    def __init__(self, parent, max_size=(360, 300)):
        super().__init__(parent)
        self.left = ImagePreview(self, "Oryginał", max_size)
        self.left.pack(side=tk.LEFT, padx=6, pady=6)
        self.right = ImagePreview(self, "Wynik", max_size)
        self.right.pack(side=tk.LEFT, padx=6, pady=6)

    def show(self, orig, res):
        self.left.show(orig)
        self.right.show(res)
