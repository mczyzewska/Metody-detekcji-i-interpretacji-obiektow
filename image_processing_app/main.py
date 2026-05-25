"""
Aplikacja do Przetwarzania Obrazów
Autorzy: Magdalena Czyżewska 21227, Adrian Witów 21319
Metody Detekcji i Interpretacji Obiektów 2025/2026
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
from gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.title("Aplikacja do Przetwarzania Obrazów – MDiIO 2025/2026")
    root.geometry("1200x800")
    root.minsize(900, 600)

    style = ttk.Style(root)
    for theme in ("clam", "aqua", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break

    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
