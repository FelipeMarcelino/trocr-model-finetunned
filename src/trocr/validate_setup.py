import torch

from trocr.config import MAX_TARGET_LENGTH
from trocr.dataset import HandwritingDataset, load_and_split_data
from trocr.model import initialize_model


def validate_model_setup():
    print("🔍 Validando configuração do modelo...")

    model, processor = initialize_model(use_peft=False)
    train_df, _ = load_and_split_data()
    dataset = HandwritingDataset(train_df, processor, MAX_TARGET_LENGTH, is_train=False)

    # Testar um batch
    sample = dataset[0]
    pixel_values = sample["pixel_values"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    print(f"✅ Pixel values shape: {pixel_values.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Vocab size: {len(processor.tokenizer)}")
    print(f"✅ PAD token ID: {processor.tokenizer.pad_token_id}")
    print(f"✅ BOS token ID: {processor.tokenizer.bos_token_id}")
    print(f"✅ EOS token ID: {processor.tokenizer.eos_token_id}")

    # Testar forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
        print(f"✅ Loss: {outputs.loss.item():.4f}")

    # Testar geração
    generated = model.generate(pixel_values, max_length=50)
    pred_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"✅ Geração inicial: '{pred_text}'")

    if len(pred_text) == 0 or pred_text == processor.tokenizer.pad_token:
        print("⚠️  AVISO: Modelo não está gerando texto válido!")

if __name__ == "__main__":
    validate_model_setup()
