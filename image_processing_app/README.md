# 🖼 Aplikacja do Przetwarzania Obrazów

> **Projekt 1** – Metody Detekcji i Interpretacji Obiektów 2025/2026  
> Akademia Nauk Stosowanych w Elblągu

**Autorzy:**
- Magdalena Czyżewska (nr indeksu: 21227)
- Adrian Witów (nr indeksu: 21319)

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Wymagania systemowe](#-wymagania-systemowe)
- [Instalacja](#-instalacja)
- [Uruchomienie](#-uruchomienie)
- [Struktura projektu](#-struktura-projektu)
- [Opis algorytmów](#-opis-algorytmów)
- [Harmonogram](#-harmonogram)

---

## 📌 Opis projektu

Desktopowa aplikacja GUI (Tkinter) do przetwarzania i analizy obrazów, łącząca klasyczne algorytmy wizji komputerowej z nowoczesnymi modelami głębokiego uczenia.

---

## ✅ Funkcjonalności

### Funkcjonalności podstawowe (Milestone 1 – 18.05.2026)

| Symbol | Opis |
|--------|------|
| **A** | Wczytywanie i zapis obrazów (PNG, JPG, BMP, TIFF, WebP) |
| **B** | Wykrywanie krawędzi – **Canny**, **Sobel**, **Laplacian** |
| **C** | Progowanie – **Otsu**, **Adaptacyjne Mean**, **Adaptacyjne Gaussian** |
| **D** | Wsadowa zmiana rozdzielczości wszystkich obrazów w folderze |
| **E** | Wsadowe dodawanie znaku wodnego (pozycja, przezroczystość, czcionka) |
| **F** | Przetwarzanie siecią neuronową – MobileNetV2 / DeepLab v3 / YOLOv5 |

### Dodatkowe funkcjonalności

| Symbol | Opis |
|--------|------|
| **III-A** | Wykrywanie szkieletu człowieka w konkretnych pozach (MediaPipe Pose, 33 punkty kluczowe) |
| **III-B** | Wykrywanie i klasyfikacja znaków drogowych (43 klasy GTSRB) |
| **III-C** | Wykrywanie mimiki twarzy i klasyfikacja emocji (DeepFace / MediaPipe Face Mesh) |

---

## 💻 Wymagania systemowe

- **Python** 3.10 lub nowszy
- **macOS** 12+, **Windows** 10/11 lub **Linux** (Ubuntu 20.04+)
- Minimum **4 GB RAM** (8 GB zalecane dla modeli DL)
- Opcjonalnie: karta graficzna NVIDIA z CUDA dla szybszego przetwarzania

---

## 🚀 Instalacja

### Krok 1 – Klonowanie repozytorium

```bash
git clone https://github.com/TWOJE_KONTO/image-processing-app.git
cd image-processing-app
```

### Krok 2 – Tworzenie wirtualnego środowiska

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Krok 3 – Instalacja zależności podstawowych

```bash
pip install Pillow opencv-python numpy
```

### Krok 4 – Instalacja PyTorch

**macOS (Apple Silicon M1/M2/M3):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**macOS (Intel) / Linux CPU:**
```bash
pip install torch torchvision
```

**Windows / Linux z GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Krok 5 – Instalacja bibliotek dla funkcji zaawansowanych

```bash
# Detekcja szkieletu
pip install mediapipe

# Detekcja emocji
pip install deepface tf-keras

# Detekcja obiektów (YOLOv5/v8)
pip install ultralytics
```

### Krok 6 – Instalacja wszystkiego na raz (opcjonalnie)

```bash
pip install -r requirements.txt
```

> **Uwaga dla macOS:** Jeśli pojawi się błąd z `tkinter`, zainstaluj go przez:
> ```bash
> brew install python-tk
> ```

---

## ▶️ Uruchomienie

```bash
# Upewnij się, że wirtualne środowisko jest aktywne
source venv/bin/activate  # macOS/Linux
# lub
venv\Scripts\activate     # Windows

# Uruchom aplikację
python main.py
```

### Uruchomienie w Visual Studio Code

1. Otwórz folder projektu: `File → Open Folder`
2. Wybierz interpreter Python (Cmd+Shift+P → `Python: Select Interpreter`) → wskaż `venv`
3. Naciśnij **F5** lub kliknij **Run → Run Without Debugging**
4. Lub w terminalu VSCode: `python main.py`

---

## 📁 Struktura projektu

```
image_processing_app/
│
├── main.py                        # Punkt wejścia aplikacji
│
├── gui/                           # Warstwa interfejsu graficznego
│   ├── __init__.py
│   ├── main_window.py             # Główne okno z zakładkami
│   ├── tab_basic.py               # Zakładki A–F (podstawowe)
│   ├── tab_advanced.py            # Zakładki III-A, III-B, III-C
│   └── image_preview.py           # Widget podglądu obrazu
│
├── processing/                    # Logika przetwarzania
│   ├── __init__.py
│   ├── edge_detection.py          # B) Canny, Sobel, Laplacian
│   ├── thresholding.py            # C) Otsu, Adaptive Mean/Gaussian
│   ├── batch_resize.py            # D) Wsadowa zmiana rozdzielczości
│   ├── watermark.py               # E) Znak wodny wsadowy
│   ├── neural_network.py          # F) MobileNetV2, DeepLab, YOLO
│   ├── skeleton_detection.py      # III-A) MediaPipe Pose
│   ├── traffic_signs.py           # III-B) Detekcja znaków (GTSRB)
│   └── facial_expression.py       # III-C) DeepFace / MediaPipe
│
├── requirements.txt               # Zależności projektu
└── README.md                      # Dokumentacja
```

---

## 🔬 Opis algorytmów

### B) Wykrywanie krawędzi

#### Canny
Wieloetapowy algorytm: rozmycie Gaussowskie → obliczenie gradientu (Sobel) → Non-Maximum Suppression → podwójne progowanie → śledzenie krawędzi przez histerezę. Parametry: `próg dolny`, `próg górny`.

#### Sobel
Oblicza gradient obrazu w kierunkach X i Y przy użyciu jąder konwolucji 3×3. Wynikowa amplituda: `sqrt(Gx² + Gy²)`. Dobry do wykrywania krawędzi z kierunkiem.

#### Laplacian
Oblicza drugą pochodną obrazu (laplacjan). Wykrywa obszary szybkich zmian intensywności. Poprzedzony rozmyciem Gaussowskim dla redukcji szumu.

---

### C) Progowanie

#### Globalne (Otsu)
Automatycznie wyznacza optymalny próg globalny, minimalizując wariancję wewnątrzklasową. Najlepszy dla obrazów z bimodalnym histogramem.

#### Adaptacyjne Mean
Oblicza lokalny próg jako średnią arytmetyczną sąsiedztwa o rozmiarze `block_size × block_size` minus stała `C`. Radzi sobie z nierównomiernym oświetleniem.

#### Adaptacyjne Gaussian
Jak Mean, ale używa ważonej sumy Gaussowskiej zamiast zwykłej średniej. Daje wygładzone, mniej szumowe wyniki.

---

### III-A) Szkielet człowieka – MediaPipe Pose

Model detekuje 33 punkty kluczowe ciała (ramiona, łokcie, nadgarstki, biodra, kolana, kostki itp.) wraz z ich wizibalnością. Klasyfikacja pozy oparta na kątach stawów:
- **Stojący**: kąt kolana > 160°
- **Siedzący**: kąt kolana 80–120°
- **Kucający**: kąt kolana 120–145°
- **Leżący**: mała różnica Y między barkiem a biodrem

---

### III-B) Znaki drogowe

Podejście dwuetapowe:
1. **Detekcja kandydatów**: segmentacja kolorów (czerwony/niebieski/żółty) w przestrzeni HSV + analiza konturów
2. **Klasyfikacja**: CNN trenowana na zbiorze GTSRB (43 klasy), fallback: analiza kształtu i koloru

---

### III-C) Mimika twarzy – DeepFace

DeepFace analizuje twarz przy użyciu modelu wytrenowanego na FER2013. Klasyfikuje 7 emocji: szczęście, smutek, złość, zaskoczenie, strach, wstręt, neutralność.

**MediaPipe Face Mesh** (alternatywny backend): 468 punktów kluczowych twarzy + geometryczna analiza:
- Kąt ust → uśmiech/smutek
- Otwarcie oczu → zaskoczenie/strach
- Pozycja brwi → złość

---

## 📅 Harmonogram

| Data | Etap |
|------|------|
| **18.05.2026** | Milestone 1 – Funkcjonalności podstawowe (A–F) |
| **01.06 – 08.06.2026** | Prezentacja projektu |

---

## 📦 Użyte biblioteki

| Biblioteka | Zastosowanie |
|------------|-------------|
| `Pillow` | Wczytywanie, zapis, manipulacja obrazami |
| `OpenCV` | Klasyczne algorytmy wizji, detekcja twarzy |
| `NumPy` | Obliczenia macierzowe na obrazach |
| `Tkinter` | Interfejs graficzny (wbudowany w Python) |
| `PyTorch + TorchVision` | Głębokie uczenie (MobileNet, DeepLab) |
| `MediaPipe` | Detekcja szkieletu, siatka twarzy |
| `DeepFace` | Analiza emocji twarzy |
| `Ultralytics` | Detekcja obiektów YOLOv5/v8 |

---

## ⚠️ Uwagi

- Biblioteki ML (PyTorch, MediaPipe, DeepFace) są **opcjonalne** – aplikacja posiada fallbacki oparte na OpenCV dla każdej funkcji zaawansowanej
- Pierwsze uruchomienie modeli (MobileNet, DeepLab, YOLO) wymaga pobrania wag (~50–200 MB) – potrzebne połączenie internetowe
- Na Apple Silicon (M1/M2/M3) PyTorch działa przez CPU (MPS backend może wymagać dodatkowej konfiguracji)

---

*Projekt zrealizowany w ramach kursu Metody Detekcji i Interpretacji Obiektów, ANS Elbląg 2025/2026*
