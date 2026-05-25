"""Główne okno aplikacji."""
import tkinter as tk
from tkinter import ttk
import os
from gui.tab_basic import TabLoadSave, TabEdgeDetection, TabThresholding, TabBatchResize, TabWatermark, TabNeuralNetwork
from gui.tab_advanced import TabSkeletonDetection, TabTrafficSignDetection, TabFacialExpression

class MainWindow:
    def __init__(self, root):
        self.root=root; self.current_image=None; self.current_image_path=None
        self._build()

    def _build(self):
        top=tk.Frame(self.root,bg="#2c3e50",height=52); top.pack(fill=tk.X); top.pack_propagate(False)
        tk.Label(top,text="🖼  Aplikacja do Przetwarzania Obrazów",bg="#2c3e50",fg="white",
                 font=("Helvetica",13,"bold")).pack(side=tk.LEFT,padx=15,pady=14)
        tk.Label(top,text="M. Czyżewska 21227  |  A. Witów 21319  |  MDiIO 2025/2026",
                 bg="#2c3e50",fg="#bdc3c7",font=("Helvetica",9)).pack(side=tk.RIGHT,padx=15,pady=14)

        nb=ttk.Notebook(self.root); nb.pack(fill=tk.BOTH,expand=True,padx=6,pady=6)
        shared={"root":self.root,"get_image":lambda:self.current_image,
                "get_path":lambda:self.current_image_path,"set_image":self._set_image}

        tabs=[("📂 Wczytaj/Zapis",TabLoadSave),("🔍 Krawędzie",TabEdgeDetection),
              ("⬛ Progowanie",TabThresholding),("📐 Zmiana rozdzielczości",TabBatchResize),
              ("💧 Znak wodny",TabWatermark),("🧠 Sieć neuronowa",TabNeuralNetwork),
              ("🦴 Szkielet człowieka",TabSkeletonDetection),
              ("🚦 Znaki drogowe",TabTrafficSignDetection),
              ("😊 Mimika twarzy",TabFacialExpression)]

        for title,Cls in tabs:
            f=ttk.Frame(nb); nb.add(f,text=title); Cls(f,shared)

        self.sv=tk.StringVar(value="Gotowy")
        tk.Label(self.root,textvariable=self.sv,bd=1,relief=tk.SUNKEN,anchor=tk.W,
                 bg="#ecf0f1",fg="#2c3e50",font=("Helvetica",9)).pack(side=tk.BOTTOM,fill=tk.X)

    def _set_image(self,img,path=None):
        self.current_image=img
        if path: self.current_image_path=path
        name=os.path.basename(path) if path else "bez nazwy"
        size=f"{img.size[0]}×{img.size[1]} px" if img else ""
        self.sv.set(f"Załadowany obraz: {name}   |   {size}")
