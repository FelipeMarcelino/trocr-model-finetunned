from setuptools import find_packages, setup

setup(
    name="trocr_handwriting",
    version="0.1.0",
    description="TrOCR training pipeline for custom handwriting recognition.",
    author="Felipe Marcelino",
    # Diga ao setuptools para procurar pacotes dentro do diretório 'src'
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
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
