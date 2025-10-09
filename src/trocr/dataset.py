import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from trocr.config import CSV_PATH, IMAGE_DIR, RANDOM_STATE, TRAIN_TEST_SPLIT_RATIO
from trocr.preprocess import apply_augmentations, get_data_augmentations


class HandwritingDataset(Dataset):
    def __init__(self, df, processor, max_target_length, is_train=False):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.is_train = is_train
        if self.is_train:
            self.augmentations = get_data_augmentations()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = IMAGE_DIR / row["image_path"]
        text = row["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Imagem não encontrada em: {image_path}")

        if self.is_train:
            image = apply_augmentations(image, self.augmentations)

        # Processa a imagem para o formato esperado pelo encoder
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Tokeniza o texto para o formato esperado pelo decoder
        # ✅ Usar o tokenizer corretamente
        encoding = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = encoding.input_ids.squeeze()
        # ✅ Substituir pad_token_id por -100 (ignorado no loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def load_and_split_data():
    """Carrega o CSV e divide os dados em conjuntos de treino e avaliação."""
    try:
        df = pd.read_csv(CSV_PATH,sep=";",encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de labels não encontrado em: {CSV_PATH}")

    # Verifica as colunas esperadas
    if "image_path" not in df.columns or "text" not in df.columns:
        raise ValueError("O CSV deve conter as colunas 'image_path' e 'text'.")

    train_df, test_df = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

import torch
