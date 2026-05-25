"""Zakładki A–F (funkcjonalności podstawowe)."""
import os, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gui.image_preview import DualPreview, ImagePreview
from processing.edge_detection import detect_edges
from processing.thresholding import threshold_image
from processing.batch_resize import batch_resize
from processing.watermark import batch_watermark
from processing.neural_network import run_neural_network

def _scroll(parent):
    c=tk.Canvas(parent,borderwidth=0,highlightthickness=0)
    sb=ttk.Scrollbar(parent,orient="vertical",command=c.yview)
    c.configure(yscrollcommand=sb.set); sb.pack(side=tk.RIGHT,fill=tk.Y); c.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
    inner=tk.Frame(c); wid=c.create_window((0,0),window=inner,anchor="nw")
    def _cfg(e): c.configure(scrollregion=c.bbox("all")); c.itemconfig(wid,width=c.winfo_width())
    inner.bind("<Configure>",_cfg); c.bind("<Configure>",_cfg)
    return inner

def _sec(p,t):
    f=tk.LabelFrame(p,text=t,padx=10,pady=8,font=("Helvetica",10,"bold"))
    f.pack(padx=14,pady=6,fill=tk.X); return f

def _log(tb,msg):
    tb.config(state=tk.NORMAL); tb.insert(tk.END,msg+"\n"); tb.see(tk.END); tb.config(state=tk.DISABLED)

def _logbox(parent):
    f=tk.Frame(parent); f.pack(fill=tk.BOTH,expand=True,padx=14,pady=4)
    sb=ttk.Scrollbar(f); sb.pack(side=tk.RIGHT,fill=tk.Y)
    t=tk.Text(f,height=9,state=tk.DISABLED,bg="#f4f6f7",yscrollcommand=sb.set)
    t.pack(fill=tk.BOTH,expand=True); sb.config(command=t.yview); return t

# ── A ──────────────────────────────────────────────────────────────────────────
class TabLoadSave:
    def __init__(self,parent,shared):
        self.shared=shared; inner=_scroll(parent)
        tk.Label(inner,text="A) Wczytywanie i Zapis Obrazów",font=("Helvetica",14,"bold")).pack(pady=12)
        btn=tk.Frame(inner); btn.pack(pady=8)
        ttk.Button(btn,text="📂  Wczytaj obraz",command=self._load,width=24).grid(row=0,column=0,padx=10)
        ttk.Button(btn,text="💾  Zapisz obraz",command=self._save,width=24).grid(row=0,column=1,padx=10)
        self.info=tk.Label(inner,text="Brak załadowanego obrazu",fg="#7f8c8d",font=("Helvetica",10)); self.info.pack(pady=4)
        self.prev=ImagePreview(inner,label="Podgląd",max_size=(640,460)); self.prev.pack(pady=10)

    def _load(self):
        path=filedialog.askopenfilename(title="Wybierz obraz",
            filetypes=[("Obrazy","*.png *.jpg *.jpeg *.bmp *.tiff *.gif *.webp"),("Wszystkie","*.*")])
        if not path: return
        try:
            from PIL import Image
            img=Image.open(path).convert("RGB"); self.shared["set_image"](img,path)
            self.info.config(text=f"✅  {os.path.basename(path)}   |   {img.size[0]}×{img.size[1]} px",fg="#27ae60")
            self.prev.show(img)
        except Exception as e: messagebox.showerror("Błąd",str(e))

    def _save(self):
        img=self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu","Najpierw wczytaj obraz."); return
        path=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp"),("TIFF","*.tiff")])
        if path:
            try: img.save(path); messagebox.showinfo("Zapisano",path)
            except Exception as e: messagebox.showerror("Błąd",str(e))

# ── B ──────────────────────────────────────────────────────────────────────────
class TabEdgeDetection:
    ALGOS=["Canny","Sobel","Laplacian"]
    def __init__(self,parent,shared):
        self.shared=shared; self._res=None; inner=_scroll(parent)
        tk.Label(inner,text="B) Wykrywanie Krawędzi",font=("Helvetica",14,"bold")).pack(pady=12)
        ctrl=_sec(inner,"Parametry")
        tk.Label(ctrl,text="Algorytm:").grid(row=0,column=0,sticky=tk.W,pady=3)
        self.algo=tk.StringVar(value="Canny")
        ttk.Combobox(ctrl,textvariable=self.algo,values=self.ALGOS,state="readonly",width=18).grid(row=0,column=1,sticky=tk.W,padx=8)
        tk.Label(ctrl,text="Próg dolny (Canny):").grid(row=1,column=0,sticky=tk.W,pady=3)
        self.low=tk.IntVar(value=50); ttk.Spinbox(ctrl,from_=0,to=500,textvariable=self.low,width=9).grid(row=1,column=1,sticky=tk.W,padx=8)
        tk.Label(ctrl,text="Próg górny (Canny):").grid(row=2,column=0,sticky=tk.W,pady=3)
        self.high=tk.IntVar(value=150); ttk.Spinbox(ctrl,from_=0,to=500,textvariable=self.high,width=9).grid(row=2,column=1,sticky=tk.W,padx=8)
        ttk.Button(ctrl,text="▶  Wykryj krawędzie",command=self._run).grid(row=3,column=0,columnspan=2,pady=10)
        self.st=tk.Label(inner,text="",fg="#2980b9"); self.st.pack()
        self.dual=DualPreview(inner); self.dual.pack(pady=6)
        ttk.Button(inner,text="💾  Zapisz wynik",command=self._save).pack(pady=4)

    def _run(self):
        img=self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu","Najpierw wczytaj obraz w zakładce A."); return
        try:
            r=detect_edges(img,self.algo.get(),{"low":self.low.get(),"high":self.high.get()})
            self._res=r; self.dual.show(img,r); self.st.config(text=f"✅  {self.algo.get()} – zakończono",fg="#27ae60")
        except Exception as e: messagebox.showerror("Błąd",str(e))

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku","Najpierw wykonaj detekcję."); return
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano",path)

# ── C ──────────────────────────────────────────────────────────────────────────
class TabThresholding:
    ALGOS=["Globalne (Otsu)","Adaptacyjne (Mean)","Adaptacyjne (Gaussian)"]
    def __init__(self,parent,shared):
        self.shared=shared; self._res=None; inner=_scroll(parent)
        tk.Label(inner,text="C) Progowanie Obrazu",font=("Helvetica",14,"bold")).pack(pady=12)
        ctrl=_sec(inner,"Parametry")
        tk.Label(ctrl,text="Algorytm:").grid(row=0,column=0,sticky=tk.W,pady=3)
        self.algo=tk.StringVar(value="Globalne (Otsu)")
        ttk.Combobox(ctrl,textvariable=self.algo,values=self.ALGOS,state="readonly",width=28).grid(row=0,column=1,sticky=tk.W,padx=8)
        tk.Label(ctrl,text="Rozmiar bloku (adapt.):").grid(row=1,column=0,sticky=tk.W,pady=3)
        self.bs=tk.IntVar(value=11); ttk.Spinbox(ctrl,from_=3,to=99,increment=2,textvariable=self.bs,width=9).grid(row=1,column=1,sticky=tk.W,padx=8)
        tk.Label(ctrl,text="Stała C (adapt.):").grid(row=2,column=0,sticky=tk.W,pady=3)
        self.C=tk.IntVar(value=2); ttk.Spinbox(ctrl,from_=0,to=50,textvariable=self.C,width=9).grid(row=2,column=1,sticky=tk.W,padx=8)
        ttk.Button(ctrl,text="▶  Proguj",command=self._run).grid(row=3,column=0,columnspan=2,pady=10)
        self.st=tk.Label(inner,text="",fg="#2980b9"); self.st.pack()
        self.dual=DualPreview(inner); self.dual.pack(pady=6)
        ttk.Button(inner,text="💾  Zapisz wynik",command=self._save).pack(pady=4)

    def _run(self):
        img=self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu","Najpierw wczytaj obraz w zakładce A."); return
        try:
            r=threshold_image(img,self.algo.get(),{"block_size":self.bs.get(),"C":self.C.get()})
            self._res=r; self.dual.show(img,r); self.st.config(text=f"✅  {self.algo.get()}",fg="#27ae60")
        except Exception as e: messagebox.showerror("Błąd",str(e))

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku","Najpierw wykonaj progowanie."); return
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano",path)

# ── D ──────────────────────────────────────────────────────────────────────────
class TabBatchResize:
    METHODS=["LANCZOS","BICUBIC","BILINEAR","NEAREST"]
    def __init__(self,parent,shared):
        self.shared=shared; inner=_scroll(parent)
        tk.Label(inner,text="D) Zmiana Rozdzielczości – Wsadowo",font=("Helvetica",14,"bold")).pack(pady=12)
        ctrl=_sec(inner,"Ustawienia")
        def fr(lbl,var,row):
            tk.Label(ctrl,text=lbl).grid(row=row,column=0,sticky=tk.W,pady=3)
            ttk.Entry(ctrl,textvariable=var,width=38).grid(row=row,column=1,padx=6)
            ttk.Button(ctrl,text="📁",width=3,command=lambda v=var:v.set(filedialog.askdirectory() or v.get())).grid(row=row,column=2)
        self.src=tk.StringVar(); self.dst=tk.StringVar()
        fr("Folder wejściowy:",self.src,0); fr("Folder wyjściowy:",self.dst,1)
        tk.Label(ctrl,text="Szerokość (px):").grid(row=2,column=0,sticky=tk.W,pady=3)
        self.w=tk.IntVar(value=800); ttk.Spinbox(ctrl,from_=1,to=9999,textvariable=self.w,width=9).grid(row=2,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Wysokość (px):").grid(row=3,column=0,sticky=tk.W,pady=3)
        self.h=tk.IntVar(value=600); ttk.Spinbox(ctrl,from_=1,to=9999,textvariable=self.h,width=9).grid(row=3,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Zachowaj proporcje:").grid(row=4,column=0,sticky=tk.W)
        self.rat=tk.BooleanVar(value=True); ttk.Checkbutton(ctrl,variable=self.rat).grid(row=4,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Interpolacja:").grid(row=5,column=0,sticky=tk.W,pady=3)
        self.mth=tk.StringVar(value="LANCZOS")
        ttk.Combobox(ctrl,textvariable=self.mth,values=self.METHODS,state="readonly",width=14).grid(row=5,column=1,sticky=tk.W,padx=6)
        ttk.Button(ctrl,text="▶  Zmień rozdzielczość",command=self._run).grid(row=6,column=0,columnspan=3,pady=10)
        self.prog=ttk.Progressbar(inner,mode="determinate"); self.prog.pack(fill=tk.X,padx=14,pady=4)
        self.log=_logbox(inner)

    def _run(self):
        if not self.src.get() or not self.dst.get():
            messagebox.showwarning("Brak folderów","Wybierz foldery."); return
        self.log.config(state=tk.NORMAL); self.log.delete("1.0",tk.END); self.log.config(state=tk.DISABLED)
        self.prog["value"]=0
        params={"width":self.w.get(),"height":self.h.get(),"keep_ratio":self.rat.get(),"method":self.mth.get()}
        def worker():
            try:
                batch_resize(self.src.get(),self.dst.get(),params,
                             progress_cb=lambda p:self.prog.config(value=p),log_cb=lambda m:_log(self.log,m))
                _log(self.log,"✅ Zakończono."); messagebox.showinfo("Gotowe","Zmiana rozdzielczości zakończona.")
            except Exception as e: messagebox.showerror("Błąd",str(e))
        threading.Thread(target=worker,daemon=True).start()

# ── E ──────────────────────────────────────────────────────────────────────────
class TabWatermark:
    POSITIONS=["Środek","Lewy-górny","Prawy-górny","Lewy-dolny","Prawy-dolny"]
    def __init__(self,parent,shared):
        self.shared=shared; inner=_scroll(parent)
        tk.Label(inner,text="E) Dodanie Znaku Wodnego – Wsadowo",font=("Helvetica",14,"bold")).pack(pady=12)
        ctrl=_sec(inner,"Ustawienia")
        def fr(lbl,var,row):
            tk.Label(ctrl,text=lbl).grid(row=row,column=0,sticky=tk.W,pady=3)
            ttk.Entry(ctrl,textvariable=var,width=38).grid(row=row,column=1,padx=6)
            ttk.Button(ctrl,text="📁",width=3,command=lambda v=var:v.set(filedialog.askdirectory() or v.get())).grid(row=row,column=2)
        self.src=tk.StringVar(); self.dst=tk.StringVar()
        fr("Folder wejściowy:",self.src,0); fr("Folder wyjściowy:",self.dst,1)
        tk.Label(ctrl,text="Tekst:").grid(row=2,column=0,sticky=tk.W,pady=3)
        self.txt=tk.StringVar(value="© 2026"); ttk.Entry(ctrl,textvariable=self.txt,width=28).grid(row=2,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Pozycja:").grid(row=3,column=0,sticky=tk.W,pady=3)
        self.pos=tk.StringVar(value="Prawy-dolny")
        ttk.Combobox(ctrl,textvariable=self.pos,values=self.POSITIONS,state="readonly",width=16).grid(row=3,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Przezroczystość (0–255):").grid(row=4,column=0,sticky=tk.W,pady=3)
        self.alp=tk.IntVar(value=140); ttk.Spinbox(ctrl,from_=0,to=255,textvariable=self.alp,width=9).grid(row=4,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Rozmiar czcionki:").grid(row=5,column=0,sticky=tk.W,pady=3)
        self.fs=tk.IntVar(value=36); ttk.Spinbox(ctrl,from_=8,to=200,textvariable=self.fs,width=9).grid(row=5,column=1,sticky=tk.W,padx=6)
        tk.Label(ctrl,text="Kolor (#RRGGBB):").grid(row=6,column=0,sticky=tk.W,pady=3)
        self.col=tk.StringVar(value="#FFFFFF"); ttk.Entry(ctrl,textvariable=self.col,width=10).grid(row=6,column=1,sticky=tk.W,padx=6)
        ttk.Button(ctrl,text="▶  Dodaj znaki wodne",command=self._run).grid(row=7,column=0,columnspan=3,pady=10)
        self.prog=ttk.Progressbar(inner,mode="determinate"); self.prog.pack(fill=tk.X,padx=14,pady=4)
        self.log=_logbox(inner)

    def _run(self):
        if not self.src.get() or not self.dst.get():
            messagebox.showwarning("Brak folderów","Wybierz foldery."); return
        self.log.config(state=tk.NORMAL); self.log.delete("1.0",tk.END); self.log.config(state=tk.DISABLED)
        params={"text":self.txt.get(),"position":self.pos.get(),"alpha":self.alp.get(),"font_size":self.fs.get(),"color":self.col.get()}
        def worker():
            try:
                batch_watermark(self.src.get(),self.dst.get(),params,
                                progress_cb=lambda p:self.prog.config(value=p),log_cb=lambda m:_log(self.log,m))
                _log(self.log,"✅ Zakończono."); messagebox.showinfo("Gotowe","Znaki wodne dodane.")
            except Exception as e: messagebox.showerror("Błąd",str(e))
        threading.Thread(target=worker,daemon=True).start()

# ── F ──────────────────────────────────────────────────────────────────────────
class TabNeuralNetwork:
    MODELS=["Klasyfikacja (MobileNetV2 – ImageNet)","Segmentacja semantyczna (DeepLab v3)","Detekcja obiektów (YOLOv5 / Haar fallback)"]
    def __init__(self,parent,shared):
        self.shared=shared; self._res=None; inner=_scroll(parent)
        tk.Label(inner,text="F) Przetwarzanie – Sztuczna Sieć Neuronowa",font=("Helvetica",14,"bold")).pack(pady=12)
        ctrl=_sec(inner,"Parametry")
        tk.Label(ctrl,text="Model:").grid(row=0,column=0,sticky=tk.W,pady=3)
        self.model=tk.StringVar(value=self.MODELS[0])
        ttk.Combobox(ctrl,textvariable=self.model,values=self.MODELS,state="readonly",width=42).grid(row=0,column=1,padx=8)
        tk.Label(ctrl,text="Próg pewności:").grid(row=1,column=0,sticky=tk.W,pady=3)
        self.conf=tk.DoubleVar(value=0.5)
        frm=tk.Frame(ctrl); frm.grid(row=1,column=1,sticky=tk.W,padx=8)
        ttk.Scale(frm,from_=0.05,to=0.99,variable=self.conf,orient=tk.HORIZONTAL,length=200).pack(side=tk.LEFT)
        tk.Label(frm,textvariable=self.conf,width=5).pack(side=tk.LEFT,padx=4)
        ttk.Button(ctrl,text="▶  Uruchom model",command=self._run).grid(row=2,column=0,columnspan=2,pady=10)
        self.st=tk.Label(inner,text="",fg="#2980b9",font=("Helvetica",10)); self.st.pack(pady=2)
        self.dual=DualPreview(inner); self.dual.pack(pady=6)
        rf=tk.LabelFrame(inner,text="Wyniki modelu",padx=8,pady=6); rf.pack(fill=tk.X,padx=14,pady=4)
        sb2=ttk.Scrollbar(rf); sb2.pack(side=tk.RIGHT,fill=tk.Y)
        self.rt=tk.Text(rf,height=7,state=tk.DISABLED,bg="#f4f6f7",yscrollcommand=sb2.set); self.rt.pack(fill=tk.BOTH); sb2.config(command=self.rt.yview)
        ttk.Button(inner,text="💾  Zapisz wynik",command=self._save).pack(pady=6)

    def _run(self):
        img=self.shared["get_image"]()
        if img is None: messagebox.showwarning("Brak obrazu","Najpierw wczytaj obraz."); return
        self.st.config(text="⏳  Przetwarzanie…",fg="#e67e22")
        def worker():
            try:
                r,info=run_neural_network(img,self.model.get(),round(self.conf.get(),2))
                self._res=r; self.dual.show(img,r)
                self.rt.config(state=tk.NORMAL); self.rt.delete("1.0",tk.END); self.rt.insert(tk.END,info); self.rt.config(state=tk.DISABLED)
                self.st.config(text="✅  Zakończono",fg="#27ae60")
            except Exception as e: self.st.config(text=f"❌  {e}",fg="#e74c3c"); messagebox.showerror("Błąd",str(e))
        threading.Thread(target=worker,daemon=True).start()

    def _save(self):
        if self._res is None: messagebox.showwarning("Brak wyniku","Najpierw uruchom model."); return
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: self._res.save(path); messagebox.showinfo("Zapisano",path)
