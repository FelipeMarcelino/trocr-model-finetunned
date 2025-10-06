import argparse

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def debug_model_generation(model_path, image_path, original_text):
    """Carrega um modelo e realiza uma predição, mostrando a comparação de tokens.
    """
    print(f"Carregando modelo e processador de: {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = TrOCRProcessor.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"\nCarregando imagem: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERRO: Imagem não encontrada em {image_path}")
        return

    # Prepara a imagem para o modelo
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Gera os IDs dos tokens
    generated_ids = model.generate(pixel_values, max_length=128)

    # Decodifica os IDs para texto
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Tokeniza o texto original para comparação
    original_ids = processor.tokenizer(original_text, return_tensors="pt").input_ids.squeeze()

    print("\n" + "="*30)
    print("         ANÁLISE DE GERAÇÃO")
    print("="*30)
    print(f"Texto Original:   '{original_text}'")
    print(f"Texto Previsto:   '{generated_text}'")
    print("-"*30)

    print("\nAnálise de Tokens:")
    print(f"Tokens Originais: {original_ids.tolist()}")
    print(f"Tokens Gerados:   {generated_ids.squeeze().tolist()}")

    print("\nComparação de Tokens (Decodificados):")
    original_tokens = [processor.tokenizer.decode([tok_id]) for tok_id in original_ids]
    generated_tokens = [processor.tokenizer.decode([tok_id]) for tok_id in generated_ids.squeeze()]

    print(f"Tokens Originais (str): {' | '.join(original_tokens)}")
    print(f"Tokens Gerados (str):   {' | '.join(generated_tokens)}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de depuração para o TrOCR")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Caminho para o checkpoint do modelo salvo (ex: ./checkpoints/checkpoint-500).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Caminho para a imagem de teste (ex: ./data/images/minha_letra.png).",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="O texto real (ground truth) contido na imagem.",
    )

    args = parser.parse_args()
    debug_model_generation(args.model_path, args.image_path, args.text)
