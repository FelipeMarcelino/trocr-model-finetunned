from setuptools import find_packages, setup

setup(
    name="trocr_handwriting",
    version="0.1.0",
    description="TrOCR training pipeline for custom handwriting recognition.",
    author="Seu Nome",
    packages=find_packages(where="."),
    install_requires=[
        "torch",
        "torchvision",
        "transformers==4.35.2",
        "datasets",
        "evaluate",
        "scikit-learn",
        "pandas",
        "peft==0.6.2",
        "loralib",
        "tensorboard",
        "accelerate",
        "jiwer",
        "python-Levenshtein",
        "Pillow",
        "albumentations",
    ],
)
