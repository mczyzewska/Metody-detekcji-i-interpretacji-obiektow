# 🖼️ Aplikacja do Przetwarzania Obrazów

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-33%20passed-brightgreen)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

**Projekt 1 · Metody Detekcji i Interpretacji Obiektów 2025/2026**  
Akademia Nauk Stosowanych w Elblągu

| Autor | Nr indeksu |
|-------|-----------|
| Magdalena Czyżewska | 21227 |
| Adrian Witów | 21319 |

</div>

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Wymagania systemowe](#-wymagania-systemowe)
- [Instalacja na macOS](#-instalacja-na-macos)
- [Instalacja na Windows](#-instalacja-na-windows)
- [Uruchomienie w VS Code](#-uruchomienie-w-vs-code)
- [Uruchomienie testów](#-uruchomienie-testów)
- [Struktura projektu](#-struktura-projektu)
- [Opis algorytmów](#-opis-algorytmów)
- [Rozwiązywanie problemów](#-rozwiązywanie-problemów)
- [Harmonogram](#-harmonogram)

---

## 📌 Opis projektu

Desktopowa aplikacja GUI napisana w Pythonie (Tkinter) do kompleksowego przetwarzania i analizy obrazów. Łączy klasyczne algorytmy wizji komputerowej (OpenCV) z nowoczesnymi modelami głębokiego uczenia (PyTorch, MediaPipe, DeepFace).

Każda funkcja AI posiada **fallback oparty na OpenCV** – aplikacja działa poprawnie nawet bez zainstalowanych bibliotek ML.

---

## ✅ Funkcjonalności

### Funkcjonalności podstawowe

| # | Zakładka | Opis |
|---|----------|------|
| A | 📂 Wczytaj/Zapis | Wczytywanie i zapis obrazów – PNG, JPG, BMP, TIFF, WebP |
| B | 🔍 Krawędzie | Wykrywanie krawędzi: **Canny**, **Sobel**, **Laplacian** |
| C | ⬛ Progowanie | Progowanie: **Otsu**, **Adaptacyjne Mean**, **Adaptacyjne Gaussian** |
| D | 📐 Rozdzielczość | Wsadowa zmiana rozdzielczości wszystkich obrazów w folderze |
| E | 💧 Znak wodny | Wsadowe dodawanie znaku wodnego (tekst, pozycja, kolor, przezroczystość) |
| F | 🧠 Sieć neuronowa | MobileNetV2 (klasyfikacja) · DeepLab v3 (segmentacja) · YOLOv5 (detekcja) |

### Funkcjonalności dodatkowe

| # | Zakładka | Opis |
|---|----------|------|
| III-A | 🦴 Szkielet | Wykrywanie szkieletu człowieka i klasyfikacja póz (MediaPipe Pose, 33 punkty) |
| III-B | 🚦 Znaki drogowe | Wykrywanie i klasyfikacja znaków drogowych – 43 klasy GTSRB |
| III-C | 😊 Mimika twarzy | Detekcja twarzy i klasyfikacja 7 emocji (DeepFace / MediaPipe Face Mesh) |

---

## 💻 Wymagania systemowe

| | Minimum | Zalecane |
|--|---------|---------|
| **Python** | 3.10 | 3.11 / 3.12 |
| **RAM** | 4 GB | 8 GB |
| **Dysk** | 2 GB | 5 GB (modele ML) |
| **macOS** | 12 Monterey | 13 Ventura+ |
| **Windows** | 10 64-bit | 11 |

---

## 🍎 Instalacja na macOS

### Krok 1 – Zainstaluj Python 3.10+

Sprawdź aktualną wersję:
```bash
python3 --version
```

Jeśli brak lub wersja starsza niż 3.10, zainstaluj przez [Homebrew](https://brew.sh):
```bash
# Instalacja Homebrew (jeśli nie masz)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalacja Pythona 3.12
brew install python@3.12

# Instalacja tkinter (wymagane dla GUI)
brew install python-tk@3.12
```

> **Apple Silicon (M1/M2/M3)?** Upewnij się że używasz natywnego ARM Pythona:
> ```bash
> python3 -c "import platform; print(platform.machine())"
> # Powinno wyświetlić: arm64
> ```

### Krok 2 – Pobierz projekt

```bash
git clone https://github.com/TWOJE_KONTO/image-processing-app.git
cd image-processing-app
```

Lub pobierz ZIP z GitHub → `Code → Download ZIP`, rozpakuj, następnie:
```bash
cd image_processing_app
```

### Krok 3 – Utwórz wirtualne środowisko

```bash
python3 -m venv venv
source venv/bin/activate
```

Po aktywacji terminal pokazuje `(venv)` na początku linii.

### Krok 4 – Zainstaluj zależności podstawowe

```bash
pip install --upgrade pip
pip install Pillow opencv-python numpy
```

### Krok 5 – Zainstaluj PyTorch

**Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Intel Mac:**
```bash
pip install torch torchvision
```

### Krok 6 – Zainstaluj biblioteki AI (opcjonalne, zalecane)

```bash
# Detekcja szkieletu – zakładka III-A
pip install mediapipe

# Detekcja emocji – zakładka III-C
pip install deepface tf-keras

# Detekcja obiektów YOLO – zakładka F
pip install ultralytics
```

### Krok 7 – Uruchom aplikację

```bash
python main.py
```

---

## 🪟 Instalacja na Windows

### Krok 1 – Zainstaluj Python 3.10+

1. Wejdź na **[python.org/downloads](https://www.python.org/downloads/)**
2. Pobierz **Python 3.12.x** – Windows installer (64-bit)
3. Uruchom instalator i **koniecznie zaznacz** `☑ Add Python to PATH`
4. Kliknij **Install Now**

Zweryfikuj instalację – otwórz **PowerShell**:
```powershell
python --version
pip --version
```

### Krok 2 – Pobierz projekt

```powershell
git clone https://github.com/TWOJE_KONTO/image-processing-app.git
cd image-processing-app
```

Lub pobierz ZIP ręcznie, rozpakuj i otwórz folder:
```powershell
cd image_processing_app
```

### Krok 3 – Utwórz wirtualne środowisko

```powershell
python -m venv venv
venv\Scripts\activate
```

> **Błąd „execution of scripts is disabled"?** Uruchom PowerShell jako Administrator i wykonaj:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Następnie wróć do normalnego okna i aktywuj ponownie.

Po aktywacji terminal pokazuje `(venv)` na początku linii.

### Krok 4 – Zainstaluj zależności podstawowe

```powershell
pip install --upgrade pip
pip install Pillow opencv-python numpy
```

### Krok 5 – Zainstaluj PyTorch

**Windows bez karty NVIDIA (CPU):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Windows z kartą NVIDIA (CUDA 12.1):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Nie wiesz czy masz CUDA? Sprawdź komendą `nvidia-smi` w terminalu. Jeśli nie istnieje – użyj wersji CPU.

### Krok 6 – Zainstaluj biblioteki AI (opcjonalne, zalecane)

```powershell
# Detekcja szkieletu – zakładka III-A
pip install mediapipe

# Detekcja emocji – zakładka III-C
pip install deepface tf-keras

# Detekcja obiektów YOLO – zakładka F
pip install ultralytics
```

### Krok 7 – Uruchom aplikację

```powershell
python main.py
```

---

## 💡 Szybka instalacja – wszystko jednym poleceniem

**macOS (Apple Silicon):**
```bash
source venv/bin/activate && \
pip install Pillow opencv-python numpy mediapipe deepface tf-keras ultralytics && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Windows / macOS Intel / Linux (CPU):**
```bash
pip install Pillow opencv-python numpy mediapipe deepface tf-keras ultralytics torch torchvision
```

---

## 🖥️ Uruchomienie w VS Code

Kroki identyczne na macOS i Windows.

### 1. Otwórz folder projektu

```
File → Open Folder → wybierz folder image_processing_app
```

### 2. Zainstaluj rozszerzenie Python (jeśli brak)

- Skrót: `Ctrl+Shift+X` (Windows) / `Cmd+Shift+X` (macOS)
- Wyszukaj: `Python` (wydawca: Microsoft) → **Install**

### 3. Wybierz interpreter Pythona

- Skrót: `Ctrl+Shift+P` / `Cmd+Shift+P`
- Wpisz: `Python: Select Interpreter`
- Wybierz:
  - macOS: `./venv/bin/python`
  - Windows: `.\venv\Scripts\python.exe`

### 4. Uruchom aplikację

**Opcja A – klawisz F5:**
- Naciśnij `F5`
- Wybierz konfigurację **„Uruchom aplikację"**

**Opcja B – terminal wbudowany w VS Code:**

macOS:
```bash
source venv/bin/activate && python main.py
```
Windows:
```powershell
venv\Scripts\activate && python main.py
```

**Opcja C – przycisk ▷:**
- Otwórz plik `main.py`
- Kliknij przycisk **▷ Run Python File** w prawym górnym rogu

---

## 🧪 Uruchomienie testów

```bash
# Aktywuj środowisko jeśli nieaktywne
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# Zainstaluj pytest
pip install pytest

# Uruchom wszystkie testy
python -m pytest tests/ -v
```

Oczekiwany wynik:
```
collected 33 items
...
33 passed in ~1.0s
```

Opcjonalnie – raport pokrycia kodu:
```bash
pip install pytest-cov
python -m pytest tests/ -v --cov=processing --cov-report=term-missing
```

---

## 📁 Struktura projektu

```
image_processing_app/
│
├── main.py                        ← punkt wejścia – uruchom ten plik
│
├── gui/                           ← interfejs graficzny (Tkinter)
│   ├── main_window.py             ← główne okno, pasek stanu, 9 zakładek
│   ├── tab_basic.py               ← zakładki A, B, C, D, E, F
│   ├── tab_advanced.py            ← zakładki III-A, III-B, III-C
│   └── image_preview.py           ← widget podglądu obraz przed/po
│
├── processing/                    ← cała logika przetwarzania obrazów
│   ├── edge_detection.py          ← Canny · Sobel · Laplacian
│   ├── thresholding.py            ← Otsu · Adaptive Mean · Gaussian
│   ├── batch_resize.py            ← wsadowa zmiana rozdzielczości
│   ├── watermark.py               ← wsadowy znak wodny
│   ├── neural_network.py          ← MobileNetV2 · DeepLab v3 · YOLOv5
│   ├── skeleton_detection.py      ← MediaPipe Pose · klasyfikacja póz
│   ├── traffic_signs.py           ← CNN + detekcja GTSRB 43 klasy
│   └── facial_expression.py       ← DeepFace · MediaPipe Face Mesh
│
├── tests/
│   └── test_processing.py         ← 33 testy jednostkowe (pytest)
│
├── .vscode/
│   ├── launch.json                ← konfiguracja F5 w VS Code
│   └── settings.json              ← ustawienia edytora i linter
│
├── requirements.txt               ← lista zależności
├── setup.py                       ← instalacja jako pakiet (opcjonalne)
├── .gitignore
└── README.md
```

---

## 🔬 Opis algorytmów

### B) Wykrywanie krawędzi

#### Canny
Wieloetapowy algorytm (1986): rozmycie Gaussowskie → gradient (Sobel) → Non-Maximum Suppression → podwójne progowanie → śledzenie przez histerezę. Parametry: `próg dolny` (domyślnie 50), `próg górny` (domyślnie 150).

#### Sobel
Oblicza gradient obrazu w kierunkach X i Y jądrami konwolucji 3×3. Amplituda = `sqrt(Gx² + Gy²)`. Wykrywa krawędzie z informacją o kierunku.

#### Laplacian
Oblicza drugą pochodną obrazu. Wykrywa obszary gwałtownych zmian intensywności. Poprzedzony rozmyciem Gaussowskim. Zwraca wartości bezwzględne, normalizowane do 0–255.

---

### C) Progowanie

#### Globalne – Otsu
Automatycznie wyznacza optymalny próg przez minimalizację wariancji wewnątrzklasowej. Najlepszy dla obrazów z bimodalnym histogramem. Poprzedzony rozmyciem Gaussowskim.

#### Adaptacyjne – Mean
Próg = średnia arytmetyczna sąsiedztwa `block_size × block_size` − stała `C`. Działa poprawnie przy nierównomiernym oświetleniu.

#### Adaptacyjne – Gaussian
Jak Mean, ale sąsiedztwo ważone rozkładem Gaussowskim. Daje wygładsze, mniej szumowe wyniki.

---

### III-A) Szkielet – MediaPipe Pose

Wykrywa 33 punkty kluczowe ciała z ich widocznością. Klasyfikacja pozy na podstawie kątów stawów (biodro–kolano–kostka):

| Poza | Kryterium |
|------|-----------|
| Stojący | Kąt kolana > 160° |
| Siedzący | Kąt kolana 80–120° |
| Kucający | Kąt kolana 120–145° |
| Leżący | Mała różnica Y bark–biodro |

---

### III-B) Znaki drogowe

Detekcja dwuetapowa: segmentacja kandydatów przez analizę kolorów HSV (czerwony / niebieski / żółty) i analizę konturów → klasyfikacja przez CNN (43 klasy GTSRB). Fallback: reguły geometryczne (kształt + kolor).

---

### III-C) Mimika twarzy

**DeepFace:** model FER2013, 7 emocji: szczęście, smutek, złość, zaskoczenie, strach, wstręt, neutralność.

**MediaPipe Face Mesh:** 468 punktów kluczowych + geometryczna klasyfikacja: kąt ust → uśmiech/smutek, otwarcie oczu → zaskoczenie/strach, pozycja brwi → złość.

---

## 🛠️ Rozwiązywanie problemów

### macOS

| Problem | Rozwiązanie |
|---------|-------------|
| `No module named '_tkinter'` | `brew install python-tk@3.12` |
| `permission denied` przy pip | `pip install --user PAKIET` |
| MediaPipe nie działa na M1/M2 | `pip install mediapipe --no-binary mediapipe` |
| Okno aplikacji się nie otwiera | Sprawdź `echo $DISPLAY` – uruchom z terminala, nie przez Finder |

### Windows

| Problem | Rozwiązanie |
|---------|-------------|
| `python` not found | Przeinstaluj Python z zaznaczonym `Add to PATH` |
| `scripts is disabled` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `Microsoft Visual C++ required` | Zainstaluj [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| OpenCV błąd z Qt | `pip install opencv-python-headless` |

### Ogólne

| Problem | Rozwiązanie |
|---------|-------------|
| Pierwsze uruchomienie modelu trwa długo | Pobierane są wagi (~50–500 MB) – potrzebny internet |
| Brak wyników detekcji | Zmniejsz próg pewności w ustawieniach zakładki |
| Aplikacja „zamarza" podczas AI | Przetwarzanie w wątku – poczekaj chwilę |

---

## 📦 Użyte biblioteki

| Biblioteka | Wersja | Zastosowanie |
|------------|--------|-------------|
| `Pillow` | ≥10.0 | Wczytywanie, zapis, manipulacja obrazami |
| `opencv-python` | ≥4.8 | Klasyczne algorytmy wizji komputerowej |
| `numpy` | ≥1.24 | Operacje macierzowe na danych obrazu |
| `tkinter` | wbudowany | Interfejs graficzny |
| `torch` + `torchvision` | ≥2.0 | MobileNetV2, DeepLab v3 |
| `mediapipe` | ≥0.10 | Detekcja szkieletu, siatka twarzy |
| `deepface` | ≥0.0.79 | Analiza emocji twarzy |
| `ultralytics` | ≥8.0 | Detekcja obiektów YOLOv8 |

---

## 📅 Harmonogram

| Data | Etap |
|------|------|
| **18.05.2026** | Milestone 1 – Funkcjonalności podstawowe (A–F) |
| **01.06 – 08.06.2026** | Prezentacja projektu |

---

<div align="center">

*Projekt zrealizowany w ramach kursu Metody Detekcji i Interpretacji Obiektów*  
*Akademia Nauk Stosowanych w Elblągu · 2025/2026*

</div>
