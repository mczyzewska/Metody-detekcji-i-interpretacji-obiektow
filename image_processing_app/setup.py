"""
Opcjonalny setup.py – umożliwia instalację projektu jako pakietu.
Użycie: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="image-processing-app",
    version="1.0.0",
    description="Aplikacja do przetwarzania obrazów – Projekt 1 MDIO 2025/2026",
    authors="Magdalena Czyżewska, Adrian Witów",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dl": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "advanced": [
            "mediapipe>=0.10.0",
            "deepface>=0.0.79",
            "tf-keras>=2.15.0",
            "ultralytics>=8.0.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "mediapipe>=0.10.0",
            "deepface>=0.0.79",
            "tf-keras>=2.15.0",
            "ultralytics>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "image-app=main:main",
        ],
    },
)
