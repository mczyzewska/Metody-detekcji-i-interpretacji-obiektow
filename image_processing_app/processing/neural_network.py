"""F) Sieć neuronowa – MobileNetV2, DeepLab v3, YOLOv5."""
import cv2, numpy as np, os
from PIL import Image, ImageDraw, ImageFont

def run_neural_network(pil_image, model_name, confidence=0.5):
    n = model_name.lower()
    if "klasyfikacja" in n or "mobilenet" in n: return _classify(pil_image, confidence)
    elif "segmentacja" in n or "deeplab" in n:  return _segment(pil_image)
    elif "detekcja" in n or "yolo" in n:         return _detect(pil_image, confidence)
    else: raise ValueError(f"Nieznany model: {model_name}")

def _font(size=18):
    for p in ["/System/Library/Fonts/Helvetica.ttc","/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p,size)
            except: pass
    return ImageFont.load_default()

def _lbl(draw,x,y,text,font,color=(255,255,0)):
    bb=draw.textbbox((x,y),text,font=font); draw.rectangle(bb,fill=(0,0,0)); draw.text((x,y),text,fill=color,font=font)

def _classify(pil_image, confidence):
    try:
        import torch, torchvision.models as M, torchvision.transforms as T
        model=M.mobilenet_v2(weights=M.MobileNet_V2_Weights.IMAGENET1K_V1); model.eval()
        tfm=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        with torch.no_grad(): probs=torch.softmax(model(tfm(pil_image.convert("RGB")).unsqueeze(0))[0],0)
        top5p,top5i=torch.topk(probs,5)
        try:
            import urllib.request,json
            labels=json.loads(urllib.request.urlopen("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",timeout=4).read())
        except: labels=[]
        lines=[(labels[i.item()] if labels else f"klasa_{i.item()}",float(p)) for p,i in zip(top5p,top5i)]
        res=pil_image.copy().convert("RGB"); d=ImageDraw.Draw(res); f=_font(22)
        for k,(l,p) in enumerate(lines[:3]): _lbl(d,10,10+k*32,f"{l}: {p*100:.1f}%",f)
        return res,"MobileNetV2 Top-5:\n\n"+"\n".join(f"  {k+1}. {l}: {p*100:.2f}%" for k,(l,p) in enumerate(lines))
    except ImportError: return _classify_fb(pil_image)

def _classify_fb(pil_image):
    arr=np.array(pil_image.convert("RGB")).astype(float); mean=arr.mean((0,1)); r,g,b=mean
    dom="czerwony" if r>g and r>b else ("zielony" if g>r and g>b else "niebieski")
    res=pil_image.copy().convert("RGB"); d=ImageDraw.Draw(res); f=_font(18)
    _lbl(d,10,10,f"Dominuje: {dom}",f); _lbl(d,10,42,f"Jasność: {arr.mean():.1f}",f)
    return res,f"⚠️  Fallback\n\nAnaliza kolorów:\n  Dominujący: {dom}\n  R={r:.1f} G={g:.1f} B={b:.1f}"

_VOC=["tło","samolot","rower","ptak","łódź","butelka","autobus","samochód","kot","krzesło","krowa",
      "stół","pies","koń","motocykl","osoba","roślina","owca","sofa","pociąg","monitor"]
_PAL=np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],
               [64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],
               [0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]],dtype=np.uint8)

def _segment(pil_image):
    try:
        import torch, torchvision.models.segmentation as S, torchvision.transforms as T
        model=S.deeplabv3_resnet50(weights=S.DeepLabV3_ResNet50_Weights.DEFAULT); model.eval()
        tfm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        with torch.no_grad(): out=model(tfm(pil_image.convert("RGB")).unsqueeze(0))["out"][0]
        mask=out.argmax(0).numpy(); color=_PAL[mask%len(_PAL)]
        orig=np.array(pil_image.convert("RGB").resize((color.shape[1],color.shape[0])))
        blend=(orig*.5+color*.5).astype(np.uint8)
        classes=[_VOC[c] if c<len(_VOC) else f"klasa {c}" for c in np.unique(mask)]
        return Image.fromarray(blend),"DeepLab v3 – Wykryte klasy:\n  "+"\n  ".join(classes)
    except ImportError:
        arr=np.array(pil_image.convert("RGB")); flat=arr.reshape(-1,3).astype(np.float32)
        crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        _,lbl,ctr=cv2.kmeans(flat,5,None,crit,10,cv2.KMEANS_PP_CENTERS)
        return Image.fromarray(np.uint8(ctr)[lbl.flatten()].reshape(arr.shape)),"⚠️  Fallback K-Means (5 klastrów)"

def _detect(pil_image, confidence):
    try:
        import torch
        model=torch.hub.load("ultralytics/yolov5","yolov5s",pretrained=True,trust_repo=True,verbose=False)
        model.conf=confidence
        results=model(pil_image); df=results.pandas().xyxy[0]
        res=pil_image.copy().convert("RGB"); d=ImageDraw.Draw(res); f=_font(16)
        for _,row in df.iterrows():
            x1,y1,x2,y2=int(row.xmin),int(row.ymin),int(row.xmax),int(row.ymax)
            d.rectangle([x1,y1,x2,y2],outline=(255,0,0),width=3)
            _lbl(d,x1,max(0,y1-22),f"{row['name']} {row.confidence:.0%}",f,(255,80,80))
        info=f"YOLOv5 – Wykryto {len(df)} obiektów:\n\n"+"\n".join(f"  {r['name']}: {r.confidence*100:.1f}%" for _,r in df.iterrows()) if len(df) else "YOLOv5 – brak obiektów"
        return res,info
    except:
        bgr=cv2.cvtColor(np.array(pil_image.convert("RGB")),cv2.COLOR_RGB2BGR)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        faces=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml").detectMultiScale(gray,1.1,5,minSize=(30,30))
        out=bgr.copy()
        for x,y,w,h in faces:
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),3); cv2.putText(out,"Twarz",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)),f"⚠️  Fallback Haar – Twarze: {len(faces)}"
