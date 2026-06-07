"""
Aplikacja do Przetwarzania Obrazów
Metody Detekcji i Interpretacji Obiektów 2025/2026
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
from gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.title("Aplikacja do Przetwarzania Obrazów")
    root.geometry("1280x820")
    root.minsize(960, 640)
    root.configure(bg="#1a1a2e")

    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
