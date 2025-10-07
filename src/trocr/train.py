import argparse

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from trocr import config
from trocr.dataset import HandwritingDataset, load_and_split_data
from trocr.logger_config import setup_logger
from trocr.metrics import compute_metrics
from trocr.model import initialize_model


def main(args):
    # 1. Criação de diretórios e configuração do logger
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(config.LOGS_DIR)

    logger.info("="*50)
    logger.info("INICIANDO PROCESSO DE TREINAMENTO DO TrOCR")
    logger.info(f"Argumentos recebidos: {args}")
    logger.info("="*50)

    # 2. Configuração do dispositivo (GPU/CPU)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de treinamento: {device.type.upper()}")

    # 3. Inicialização do modelo e processador
    model, processor = initialize_model(use_peft=args.use_peft)
    model.to(device)

    # 4. Carregamento e preparação dos dados
    logger.info("Carregando e dividindo os dados...")
    train_df, eval_df = load_and_split_data()
    logger.info(f"Dados carregados: {len(train_df)} para treino, {len(eval_df)} para avaliação.")

    train_dataset = HandwritingDataset(train_df, processor, config.MAX_TARGET_LENGTH, is_train=True)
    eval_dataset = HandwritingDataset(eval_df, processor, config.MAX_TARGET_LENGTH, is_train=False)

    # 5. Definição dos argumentos de treinamento
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(), # Usa mixed-precision se houver GPU
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        max_grad_norm=1.0,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=config.TENSORBOARD_DIR,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
    )

    # 6. Instanciação do Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(p, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 7. Treinamento
    logger.info("Iniciando o treinamento...")
    trainer.train()
    logger.info("Treinamento concluído.")

    # 8. Salvando o melhor modelo
    logger.info(f"Salvando o melhor modelo em: {config.BEST_MODEL_DIR}")
    trainer.save_model(config.BEST_MODEL_DIR)
    processor.save_pretrained(config.BEST_MODEL_DIR)
    logger.info("Processo finalizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento do Modelo TrOCR")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Número de épocas de treinamento.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Tamanho do batch de treino e avaliação.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Taxa de aprendizado inicial.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Frequência de log (em passos).",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Frequência de avaliação e salvamento (em passos).",
    )
    parser.add_argument(
        "--use_peft", action="store_true", help="Ativa o treinamento com PEFT/LoRA.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
