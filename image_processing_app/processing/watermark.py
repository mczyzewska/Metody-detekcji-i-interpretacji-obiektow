"""E) Wsadowy znak wodny."""
import os
from PIL import Image, ImageDraw, ImageFont

SUPPORTED = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
_POS_MAP  = {"Środek":"center","Lewy-górny":"top-left","Prawy-górny":"top-right",
             "Lewy-dolny":"bottom-left","Prawy-dolny":"bottom-right"}
_FONTS = ["/System/Library/Fonts/Helvetica.ttc","/Library/Fonts/Arial.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","C:/Windows/Fonts/arial.ttf"]

def _font(size):
    for p in _FONTS:
        if os.path.exists(p):
            try: return ImageFont.truetype(p,size)
            except: pass
    return ImageFont.load_default()

def _hex_rgba(h,a):
    h=h.lstrip("#")
    if len(h)==3: h="".join(c*2 for c in h)
    return int(h[:2],16),int(h[2:4],16),int(h[4:],16),a

def _xy(iw,ih,tw,th,pos,m=18):
    d={"center":((iw-tw)//2,(ih-th)//2),"top-left":(m,m),
       "top-right":(iw-tw-m,m),"bottom-left":(m,ih-th-m),"bottom-right":(iw-tw-m,ih-th-m)}
    return d.get(pos,d["bottom-right"])

def add_watermark(img, params):
    text=params.get("text","© Watermark"); pos=_POS_MAP.get(params.get("position","Prawy-dolny"),"bottom-right")
    alpha=int(params.get("alpha",140)); fsize=int(params.get("font_size",36)); color=params.get("color","#FFFFFF")
    base=img.convert("RGBA"); layer=Image.new("RGBA",base.size,(0,0,0,0))
    draw=ImageDraw.Draw(layer); font=_font(fsize)
    bb=draw.textbbox((0,0),text,font=font); tw,th=bb[2]-bb[0],bb[3]-bb[1]
    x,y=_xy(base.width,base.height,tw,th,pos)
    draw.text((x+2,y+2),text,font=font,fill=(0,0,0,min(alpha,160)))
    draw.text((x,y),text,font=font,fill=_hex_rgba(color,alpha))
    return Image.alpha_composite(base,layer).convert("RGB")

def batch_watermark(src,dst,params,progress_cb=None,log_cb=None):
    os.makedirs(dst,exist_ok=True)
    files=[f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in SUPPORTED]
    if not files:
        if log_cb: log_cb("⚠️  Brak obrazów."); return
    if log_cb: log_cb(f"Znaleziono {len(files)} obrazów…")
    for i,name in enumerate(files,1):
        try:
            with Image.open(os.path.join(src,name)) as img:
                add_watermark(img.convert("RGB"),params).save(os.path.join(dst,name))
            if log_cb: log_cb(f"[{i}/{len(files)}] ✅ {name}")
        except Exception as e:
            if log_cb: log_cb(f"[{i}/{len(files)}] ❌ {name}: {e}")
        if progress_cb: progress_cb(int(i/len(files)*100))
