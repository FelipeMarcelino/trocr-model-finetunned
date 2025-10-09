import argparse

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from trocr import config
from trocr.dataset import HandwritingDataset, load_and_split_data
from trocr.logger_config import setup_logger
from trocr.metrics import compute_metrics
from trocr.model import initialize_model


def save_best_model(trainer, processor, base_model, logger, use_peft: bool):
    """Salva o melhor modelo encontrado. Se PEFT foi usado, mescla os pesos primeiro.
    Caso contrário, salva o modelo completo diretamente do melhor checkpoint.
    """
    logger.info("Iniciando processo para salvar o melhor modelo...")
    best_checkpoint_path = trainer.state.best_model_checkpoint

    if best_checkpoint_path is None:
        logger.warning("Nenhum 'melhor' checkpoint foi salvo durante o treinamento.")
        logger.warning("Nenhum modelo final será salvo.")
        return

    logger.info(f"O melhor checkpoint encontrado está em: {best_checkpoint_path}")
    final_model = None

    if use_peft:
        logger.info("Modo PEFT: Carregando adaptador LoRA para mesclagem...")
        try:
            # Carrega os pesos do adaptador LoRA sobre o modelo base
            peft_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
            # Mescla os pesos para criar um modelo completo
            final_model = peft_model.merge_and_unload()
            logger.info("Pesos do adaptador mesclados com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar e mesclar o PeftModel: {e}", exc_info=True)
            return
    else:
        logger.info("Modo Full Fine-Tuning: Carregando o modelo completo do checkpoint...")
        try:
            # O checkpoint já contém o modelo completo, basta carregá-lo
            final_model = VisionEncoderDecoderModel.from_pretrained(best_checkpoint_path)
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo do checkpoint: {e}", exc_info=True)
            return

    if final_model:
        logger.info(f"Salvando o modelo final e o processador em: {config.BEST_MODEL_DIR}")
        final_model.save_pretrained(config.BEST_MODEL_DIR)
        processor.save_pretrained(config.BEST_MODEL_DIR)
        logger.info("Melhor modelo salvo com sucesso!")

def log_sample_predictions(model, processor, eval_dataset, device, logger, num_samples=3):
    """Loga exemplos de predições para debug"""
    model.eval()
    logger.info("\n" + "="*50)
    logger.info("EXEMPLOS DE PREDIÇÕES:")
    logger.info("="*50)

    for i in range(min(num_samples, len(eval_dataset))):
        sample = eval_dataset[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)

        with torch.no_grad():
            generated = model.generate(pixel_values=pixel_values,max_length=config.MAX_TARGET_LENGTH)

        pred_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

        # Recupera o texto real removendo os tokens de padding (-100)
        true_labels = [l for l in sample["labels"].tolist() if l != -100]
        true_text = processor.tokenizer.decode(true_labels, skip_special_tokens=True)

        logger.info(f"\nExemplo {i+1}:")
        logger.info(f"  Real: '{true_text}'")
        logger.info(f"  Pred: '{pred_text}'")
        logger.info(f"  Match: {'✓' if pred_text.strip() == true_text.strip() else '✗'}")

    logger.info("="*50 + "\n")
    model.train()


class LogPredictionsCallback(TrainerCallback):
    """Callback customizado para logar predições durante o treinamento"""

    def __init__(self, processor, eval_dataset, device, logger,num_samples=3, log_frequency=5):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.device = device
        self.logger = logger
        self.log_frequency = log_frequency
        self.eval_counter = 0
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Chamado após cada avaliação"""
        self.eval_counter += 1

        # Loga predições a cada N avaliações
        if self.eval_counter % self.log_frequency == 0:
            log_sample_predictions(
                model=model,
                processor=self.processor,
                eval_dataset=self.eval_dataset,
                device=self.device,
                logger=self.logger,
                num_samples=self.num_samples,
            )


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

    base_model = None
    if args.use_peft:
        base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=config.ENCODER_MODEL_NAME,
        decoder_pretrained_model_name_or_path=config.DECODER_MODEL_NAME,
                )

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
        max_grad_norm=0.5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=config.TENSORBOARD_DIR,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        )

    log_callback = LogPredictionsCallback(
        processor=processor,
        eval_dataset=eval_dataset,
        device=device,
        logger=logger,
        log_frequency=args.log_pred_frequency,
        num_samples=args.num_samples_log,
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
        callbacks=[log_callback],  # Adiciona o callback
    )
# 6. Treinamento com try/finally para interrupção segura
    try:
        logger.info(f"Iniciando treinamento {'com PEFT/LoRA' if args.use_peft else 'completo'}...")
        logger.info("Pressione Ctrl+C para interromper e salvar o melhor resultado até o momento.")
        logger.info("Exemplos antes do treinamento:")
        trainer.train()
        logger.info("Treinamento concluído normalmente.")
    except KeyboardInterrupt:
        logger.warning("\nTreinamento interrompido pelo usuário (Ctrl+C).")
    finally:
        logger.info("--- FASE DE SALVAMENTO FINAL ---")
        # Passamos o argumento 'use_peft' para a função de salvamento
        save_best_model(trainer, processor, base_model, logger, use_peft=args.use_peft)

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
    parser.add_argument(
        "--log_pred_frequency", type=int, default=5,
        help="Frequência para logar predições de exemplo (a cada N avaliações).",
    )

    parser.add_argument(
        "--num_samples_log", type=int, default=5,
        help="Total de samples utilizada no log",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
