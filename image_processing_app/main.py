"""
Aplikacja do przetwarzania obrazów
Autorzy: Magdalena Czyżewska 21227, Adrian Witów 21319
Metody Detekcji i Interpretacji obiektów 2025/2026
"""

import sys
import os

# Ensure the app directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
from gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.title("Aplikacja do Przetwarzania Obrazów")
    root.geometry("1200x800")
    root.minsize(900, 600)

    # Apply a modern theme
    style = ttk.Style(root)
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")
    elif "aqua" in available_themes:
        style.theme_use("aqua")

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
