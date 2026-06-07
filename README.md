# 🖼️ Aplikacja do Przetwarzania Obrazów

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey)
![Tests](https://img.shields.io/badge/Testy-31%20passed-brightgreen)

**Metody Detekcji i Interpretacji Obiektów 2025/2026**
Akademia Nauk Stosowanych w Elblągu

| Autor | Nr indeksu |
|-------|-----------|
| Witów Adrian | 21319 |
| Magdalena Czyżewska | 21227 |

</div>

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Technologie i modele](#-technologie-i-modele)
- [Instalacja – macOS](#-instalacja--macos)
- [Instalacja – Windows](#-instalacja--windows)
- [Uruchomienie w VS Code](#-uruchomienie-w-vs-code)
- [Szczegóły zakładek](#-szczegóły-zakładek)
- [Testy jednostkowe](#-testy-jednostkowe)
- [Struktura projektu](#-struktura-projektu)
- [Rozwiązywanie problemów](#-rozwiązywanie-problemów)

---

## 📌 Opis projektu

Desktopowa aplikacja GUI napisana w Pythonie z ciemnym motywem graficznym (Tkinter). Łączy klasyczne algorytmy wizji komputerowej z nowoczesnymi modelami głębokiego uczenia. Zawiera **9 zakładek** pokrywających pełen zakres wymagań projektowych – od podstawowego wczytywania obrazów po detekcję szkieletu człowieka i analizę emocji twarzy.

Każda funkcja AI posiada **fallback oparty na OpenCV** – aplikacja działa w pełni bez zainstalowanych bibliotek ML.

---

## ✅ Funkcjonalności

### Zakładki podstawowe (Milestone 1)

| Zakładka | Symbol | Opis |
|----------|--------|------|
| 📂 Wczytaj / Zapis | A | Wczytywanie i zapis obrazów – PNG, JPG, BMP, TIFF, WebP |
| 🔍 Krawędzie | B | Wykrywanie krawędzi: **Canny**, **Sobel**, **Laplacian** |
| ⬛ Progowanie | C | Progowanie: **Otsu**, **Adaptacyjne Mean**, **Adaptacyjne Gaussian** |
| 📐 Rozdzielczość | D | Wsadowa zmiana rozdzielczości wszystkich obrazów w folderze |
| 💧 Znak wodny | E | Wsadowe dodawanie tekstowego znaku wodnego do folderu obrazów |
| 🧠 Sieć neuronowa | F | Klasyfikacja (MobileNetV2) · Segmentacja (DeepLab v3) · Detekcja (YOLOv5) |

### Zakładki zaawansowane

| Zakładka | Symbol | Opis |
|----------|--------|------|
| 🦴 Szkielet człowieka | III-A | Detekcja 33 punktów kluczowych ciała + klasyfikacja pozy (MediaPipe) |
| 🌸 Filtr różowy | III-B | 5 trybów filtra różowego z regulowaną intensywnością i winiетowaniem |
| 😊 Mimika twarzy | III-C | Detekcja twarzy + klasyfikacja 7 emocji (DeepFace / MediaPipe) |

---

## 🛠️ Technologie i modele

### Biblioteki

| Biblioteka | Wersja | Zastosowanie |
|------------|--------|-------------|
| `Pillow` | ≥ 10.0 | Wczytywanie, zapis, manipulacja obrazami, znak wodny |
| `opencv-python` | ≥ 4.8 | Algorytmy krawędzi, progowania, Haar Cascades (fallback) |
| `numpy` | ≥ 1.24 | Operacje macierzowe, filtry różowego |
| `tkinter` | wbudowany | Interfejs graficzny GUI – ciemny motyw |
| `torch` + `torchvision` | ≥ 2.0 | Modele głębokiego uczenia |
| `mediapipe` | ≥ 0.10 | Detekcja szkieletu i siatki twarzy (Tasks API) |
| `deepface` + `tf-keras` | ≥ 0.0.79 | Analiza emocji twarzy (FER2013) |
| `ultralytics` | ≥ 8.0 | Detekcja obiektów YOLOv5/v8 |

### Modele AI

| Model | Zadanie | Zbiór danych | Rozmiar |
|-------|---------|-------------|---------|
| **MobileNetV2** | Klasyfikacja obrazu | ImageNet (1000 klas) | ~14 MB |
| **DeepLab v3 ResNet-50** | Segmentacja semantyczna | PASCAL VOC (21 klas) | ~160 MB |
| **YOLOv5s** | Detekcja obiektów | COCO (80 klas) | ~28 MB |
| **MediaPipe PoseLandmarker Full** | Detekcja szkieletu | Własny Google | ~29 MB |
| **MediaPipe FaceLandmarker** | Siatka twarzy (468 pkt.) | Własny Google | ~6 MB |
| **DeepFace FER2013** | Klasyfikacja emocji | FER2013 (7 emocji) | ~54 MB |

> Modele MediaPipe pobierane są automatycznie przy pierwszym użyciu z Google Storage.

---

## 🍎 Instalacja – macOS

### Krok 1 – Python 3.10+

```bash
python3 --version   # sprawdź wersję
```

Jeśli brak lub < 3.10:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
brew install python-tk@3.12   # wymagane dla GUI
```

> **Apple Silicon (M1/M2/M3):** sprawdź `python3 -c "import platform; print(platform.machine())"` → powinno wyświetlić `arm64`

### Krok 2 – Pobierz projekt

```bash
git clone https://github.com/TWOJE_KONTO/image-processing-app.git
cd image-processing-app
```

### Krok 3 – Wirtualne środowisko

```bash
python3 -m venv venv
source venv/bin/activate
```

### Krok 4 – Zależności podstawowe

```bash
pip install --upgrade pip
pip install Pillow opencv-python numpy
```

### Krok 5 – PyTorch

**Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Intel Mac:**
```bash
pip install torch torchvision
```

### Krok 6 – Biblioteki AI (opcjonalne)

```bash
pip install mediapipe
pip install deepface tf-keras
pip install ultralytics
```

### Krok 7 – Uruchom

```bash
python main.py
```

---

## 🪟 Instalacja – Windows

### Krok 1 – Python 3.10+

1. Pobierz z [python.org/downloads](https://www.python.org/downloads/) → Python 3.12
2. Zaznacz `☑ Add Python to PATH`
3. Kliknij **Install Now**

```powershell
python --version   # weryfikacja
```

### Krok 2 – Pobierz projekt

```powershell
git clone https://github.com/TWOJE_KONTO/image-processing-app.git
cd image-processing-app
```

### Krok 3 – Wirtualne środowisko

```powershell
python -m venv venv
venv\Scripts\activate
```

> Błąd `scripts is disabled`? Uruchom: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Krok 4 – Zależności podstawowe

```powershell
pip install --upgrade pip
pip install Pillow opencv-python numpy
```

### Krok 5 – PyTorch

**CPU (bez karty NVIDIA):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.1):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Krok 6 – Biblioteki AI (opcjonalne)

```powershell
pip install mediapipe
pip install deepface tf-keras
pip install ultralytics
```

### Krok 7 – Uruchom

```powershell
python main.py
```

---

## 🖥️ Uruchomienie w VS Code

1. `File → Open Folder` → wybierz folder `image_processing_app`
2. `Ctrl/Cmd+Shift+P` → `Python: Select Interpreter` → wybierz `venv`
3. Naciśnij **F5** → konfiguracja `Uruchom aplikację`

---

## 🔬 Szczegóły zakładek

### A) Wczytywanie i Zapis

- Obsługiwane formaty wejściowe: **PNG, JPG/JPEG, BMP, TIFF, GIF, WebP**
- Podgląd wczytanego obrazu z informacją o rozmiarze i trybie
- Zapis do wybranego formatu z oknem dialogowym
- Aktywny obraz przekazywany do wszystkich zakładek B, C, F, III-A, III-B, III-C

---

### B) Wykrywanie Krawędzi

#### Canny (John Canny, 1986)
Wieloetapowy algorytm – uznawany za standard w detekcji krawędzi:
1. **Rozmycie Gaussowskie** (jądro 5×5) – redukcja szumu
2. **Gradient intensywności** – operator Sobela w osiach X i Y
3. **Non-Maximum Suppression** – ścieńczenie krawędzi do 1 piksela
4. **Podwójne progowanie** – klasyfikacja: pewne / kandydaci / tło
5. **Hystereza** – śledzenie połączeń od krawędzi pewnych

Parametry: `próg dolny` (0–500), `próg górny` (0–500)

#### Sobel
Oblicza gradient obrazu przy użyciu jąder konwolucji 3×3:
```
Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
Gy = [[-1,-2,-1],  [ 0, 0,  0], [+1,+2,+1]]
Amplituda = sqrt(Gx² + Gy²)  → normalizacja 0–255
```

#### Laplacian
Oblicza drugą pochodną obrazu. Poprzedzony rozmyciem Gaussowskim (3×3). Zwraca wartości bezwzględne, normalizowane do 0–255.

---

### C) Progowanie

#### Globalne – Otsu
Automatycznie minimalizuje wariancję wewnątrzklasową. Poprzedzony rozmyciem Gaussowskim.

#### Adaptacyjne – Mean
```
T(x,y) = mean(sąsiedztwo block_size×block_size) − C
```

#### Adaptacyjne – Gaussian
```
T(x,y) = Σ[w_gauss(i,j) · I(x+i, y+j)] − C
```
Gładsze wyniki niż Mean dzięki ważeniu Gaussowskiemu.

---

### D) Zmiana Rozdzielczości (Wsadowo)

- Przetwarza wszystkie obrazy (`jpg`, `png`, `bmp`, `tiff`, `webp`) w wybranym folderze
- Opcja zachowania proporcji (`thumbnail`) lub skalowania do dokładnych wymiarów (`resize`)
- Metody interpolacji: **LANCZOS** (najlepsza jakość), **BICUBIC**, **BILINEAR**, **NEAREST**
- Pasek postępu + log operacji w czasie rzeczywistym

---

### E) Znak Wodny (Wsadowo)

- Nakłada tekstowy znak wodny na wszystkie obrazy w folderze
- Parametry: treść tekstu, pozycja (5 opcji: środek / 4 rogi), przezroczystość (0–255), rozmiar czcionki, kolor (#RRGGBB)
- Subtelny cień tekstu (2 px offset) dla czytelności na jasnych i ciemnych tłach
- Implementacja: osobna warstwa `RGBA` + `alpha_composite`

---

### F) Sieć Neuronowa

#### Klasyfikacja – MobileNetV2
- Architektura: Inverted Residual Blocks + Linear Bottlenecks + Depthwise Separable Convolutions
- Wejście: tensor `224×224×3`, normalizowany (μ=`[0.485, 0.456, 0.406]`, σ=`[0.229, 0.224, 0.225]`)
- Wyjście: Top-5 predykcji z prawdopodobieństwami (softmax, 1000 klas ImageNet)
- **Fallback:** analiza statystyczna kanałów R/G/B (jasność, dominujący kolor)

#### Segmentacja – DeepLab v3 (ResNet-50)
- Architektura: Atrous Convolutions + ASPP (Atrous Spatial Pyramid Pooling)
- Zbiór: PASCAL VOC – 21 klas (tło, osoba, samochód, pies, kot, rower itp.)
- Wyjście: mapa klas nałożona na oryginał (blend 50/50) z kolorową paletą klas
- **Fallback:** segmentacja K-Means (5 klastrów kolorów, OpenCV)

#### Detekcja obiektów – YOLOv5s
- Architektura: jednoetapowy detektor CNN, siatka predykcji ramek (anchor-based)
- Wariant: `yolov5s` – small (~7.2 M parametrów), szybki na CPU
- Zbiór: COCO – 80 klas (pojazdy, zwierzęta, meble, elektronika itp.)
- Wyjście: ramki otaczające z etykietami i procentem pewności
- **Fallback:** OpenCV Haar Cascades (twarze frontalne + sylwetki pełnego ciała)

---

### III-A) Szkielet Człowieka

**Model:** MediaPipe PoseLandmarker Full (Tasks API ≥ 0.10)
- 33 punkty kluczowe ciała z współrzędnymi (x, y, z) i widocznością
- 16 połączeń szkieletu (kończyny górne, dolne, tułów)
- Model pobierany automatycznie przy pierwszym użyciu (~29 MB)

**Klasyfikacja pozy** (kąty stawów biodro–kolano–kostka):

| Poza | Kryterium |
|------|-----------|
| Stojący | kąt kolana > 160° |
| Siedzący | kąt kolana 80–120° |
| Kucający | kąt kolana 120–145° |
| Leżący | mała różnica Y bark–biodro (< 0.15) |
| W ruchu | pozostałe przypadki |

**Fallback:** OpenCV Haar Cascades (pełna sylwetka + górna część ciała)

---

### III-B) Filtr Różowy 🌸

Konwertuje obraz na odcienie różowego w 5 trybach:

| Tryb | Opis | Technika |
|------|------|----------|
| **Klasyczny różowy** | Ciepły, naturalny róż | Boosted R channel + obniżone G/B |
| **Ciepły róż** | Kremowo-różowy | R +80, G/B tłumione |
| **Neonowy różowy** | Intensywny `#ff006e` | Desaturacja + overlay neonowy |
| **Pastelowy różowy** | Jasny, stonowany | Rozjaśnienie + blend `light pink (255,182,193)` |
| **Malinowy** | Głęboki ciemny róż | R nasycony, G/B mocno tłumione |

Dodatkowe opcje: **intensywność** (0.05–1.0, suwak), **rozmycie Gaussowskie** (radius 1.5), **winietowanie** (ciemne krawędzie wzmacniające efekt).

---

### III-C) Mimika Twarzy

**Backend 1 – DeepFace (FER2013)**
- Model wytrenowany na zbiorze FER2013 (~35 000 zdjęć twarzy)
- Klasyfikuje 7 emocji: **szczęście, smutek, złość, zaskoczenie, strach, wstręt, neutralność**
- Zwraca rozkład procentowy wszystkich emocji dla każdej twarzy

**Backend 2 – MediaPipe FaceLandmarker (Tasks API ≥ 0.10)**
- 468 punktów kluczowych twarzy + 52 współczynniki blendshapes
- Emocje wyznaczane geometrycznie z blendshapes:
  - `mouthSmileLeft/Right` → szczęście
  - `mouthFrownLeft/Right` → smutek
  - `browDownLeft/Right` → złość
  - `jawOpen + eyeWide` → zaskoczenie / strach
  - `noseSneerLeft` → wstręt
- Model pobierany automatycznie (~6 MB)

**Fallback:** OpenCV Haar Cascades (detekcja twarzy) + `haarcascade_smile.xml` (uśmiech)

---

## 🧪 Testy jednostkowe

```bash
source venv/bin/activate    # macOS
venv\Scripts\activate       # Windows

pip install pytest
python -m pytest tests/ -v
```

Pokrycie testów: **31 testów** obejmujących:
- Detekcja krawędzi (Canny, Sobel, Laplacian)
- Progowanie (Otsu, Mean, Gaussian)
- Wsadowa zmiana rozdzielczości (proporcje, dokładny rozmiar, pasek postępu)
- Znak wodny (wszystkie pozycje, zmiana pikseli, batch)
- Filtr różowy (wszystkie 5 trybów, NMS, klasyfikacja kształtu)
- Szkielet człowieka (fallback OpenCV)
- Mimika twarzy (fallback, blendshapes=None)
- Sieci neuronowe (wszystkie 3 fallbacki)

---

## 📁 Struktura projektu

```
image_processing_app/
│
├── main.py                        ← uruchom ten plik
│
├── gui/
│   ├── main_window.py             ← okno, ciemny motyw ttk, 9 zakładek
│   ├── tab_basic.py               ← zakładki A–F
│   ├── tab_advanced.py            ← zakładki III-A, III-B, III-C
│   └── image_preview.py           ← DualPreview (500×380), ciemne kanwasy
│
├── processing/
│   ├── edge_detection.py          ← Canny · Sobel · Laplacian
│   ├── thresholding.py            ← Otsu · Adaptive Mean · Gaussian
│   ├── batch_resize.py            ← wsadowa zmiana rozdzielczości
│   ├── watermark.py               ← wsadowy znak wodny (RGBA layer)
│   ├── neural_network.py          ← MobileNetV2 · DeepLab v3 · YOLOv5
│   ├── skeleton_detection.py      ← MediaPipe PoseLandmarker (Tasks API)
│   ├── traffic_signs.py           ← 5 trybów filtra różowego + winietowanie
│   └── facial_expression.py       ← DeepFace + MediaPipe FaceLandmarker
│
├── tests/
│   └── test_all.py                ← 31 testów jednostkowych (pytest)
│
├── .vscode/
│   ├── launch.json
│   └── settings.json
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Rozwiązywanie problemów

### macOS

| Problem | Rozwiązanie |
|---------|-------------|
| `No module named '_tkinter'` | `brew install python-tk@3.12` |
| Okno nie otwiera się | Uruchom z terminala, nie przez Finder |
| MediaPipe – błąd M1/M2 | `pip install mediapipe --no-binary mediapipe` |
| `module 'mediapipe' has no attribute 'solutions'` | Zaktualizuj: `pip install mediapipe --upgrade` (wymagane ≥ 0.10) |

### Windows

| Problem | Rozwiązanie |
|---------|-------------|
| `python` not found | Przeinstaluj z zaznaczonym `Add to PATH` |
| `scripts is disabled` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `Microsoft Visual C++ required` | Zainstaluj [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

### Ogólne

| Problem | Rozwiązanie |
|---------|-------------|
| Pierwsze uruchomienie AI trwa długo | Pobierane wagi (~6–160 MB) – wymagany internet |
| Suwak pewności skacze co 1 | Używaj wersji v4+ (naprawiono `resolution=0.05`) |
| DeepFace błąd TF | `pip install tf-keras` |

---

<div align="center">

**Prezentacja projektu: 01.06 – 08.06.2026**

*Metody Detekcji i Interpretacji Obiektów · ANS Elbląg · 2025/2026*

</div>
