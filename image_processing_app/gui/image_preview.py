"""Widgety podglądu obrazów – większe, ciemny motyw."""
import tkinter as tk
from PIL import Image, ImageTk

C_BG      = "#1a1a2e"
C_CANVAS  = "#0f3460"
C_BORDER  = "#e94560"
C_TEXT    = "#a0a0b0"


class ImagePreview(tk.LabelFrame):
    def __init__(self, parent, label="Podgląd", max_size=(520, 400)):
        super().__init__(
            parent, text=label,
            bg=C_BG, fg=C_BORDER,
            font=("Helvetica", 9, "bold"),
            bd=1, relief="flat",
            padx=2, pady=2,
            highlightbackground=C_BORDER,
            highlightthickness=1,
        )
        self.max_size = max_size
        self._tk = None

        self._canvas = tk.Canvas(
            self,
            width=max_size[0],
            height=max_size[1],
            bg=C_CANVAS,
            highlightthickness=0,
        )
        self._canvas.pack()

        self._placeholder = self._canvas.create_text(
            max_size[0] // 2,
            max_size[1] // 2,
            text="Brak obrazu",
            fill=C_TEXT,
            font=("Helvetica", 11),
        )
        self._image_item = None

    def show(self, pil_image):
        if pil_image is None:
            if self._image_item is not None:
                self._canvas.delete(self._image_item)
                self._image_item = None
            self._canvas.itemconfigure(self._placeholder, state="normal")
            self._tk = None
            return

        img = pil_image.copy()
        img.thumbnail(self.max_size, Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(img)

        cx = self.max_size[0] // 2
        cy = self.max_size[1] // 2

        if self._image_item is None:
            self._image_item = self._canvas.create_image(cx, cy, anchor="center", image=self._tk)
        else:
            self._canvas.itemconfigure(self._image_item, image=self._tk)
            self._canvas.coords(self._image_item, cx, cy)

        self._canvas.itemconfigure(self._placeholder, state="hidden")


class DualPreview(tk.Frame):
    """Dwa podglądy obok siebie: Oryginał | Wynik."""
    def __init__(self, parent, max_size=(500, 380)):
        super().__init__(parent, bg=C_BG)
        self.left  = ImagePreview(self, "Oryginał", max_size)
        self.right = ImagePreview(self, "Wynik",    max_size)
        self.left.pack(side=tk.LEFT,  padx=8, pady=6)
        self.right.pack(side=tk.LEFT, padx=8, pady=6)

    def show(self, orig, res):
        self.left.show(orig)
        self.right.show(res)
