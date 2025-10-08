import albumentations as A
from PIL import Image


def get_data_augmentations():
    """Retorna um pipeline de augmentations para as imagens de treino."""
    return A.Compose([
        A.Rotate(limit=5, p=0.5),  # Reduzir probabilidade
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        # Adicionar novas augmentações
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=0.05, p=0.2),
    ])

def apply_augmentations(image: Image.Image, transform: A.Compose):
    """Aplica as transformações do albumentations a uma imagem PIL."""
    # Albumentations espera um array numpy
    import numpy as np
    image_np = np.array(image.convert("RGB"))
    augmented_image_np = transform(image=image_np)["image"]
    return Image.fromarray(augmented_image_np)
