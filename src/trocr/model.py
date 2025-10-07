import logging

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from trocr.config import DECODER_MODEL_NAME, LORA_ALPHA, LORA_DROPOUT, LORA_R, PROCESSOR_MODEL_NAME

logger = logging.getLogger("trocr.model")

def initialize_model(use_peft: bool = False):
    """Inicializa o modelo TrOCR, o processador e, opcionalmente, aplica PEFT/LoRA.
    """
    logger.info("Inicializando o processador TrOCR...")
    processor = TrOCRProcessor.from_pretrained(PROCESSOR_MODEL_NAME)

    logger.info("Inicializando o modelo VisionEncoderDecoder...")
    model = VisionEncoderDecoderModel.from_pretrained(PROCESSOR_MODEL_NAME)


    logger.info(f"Carregando tokenizer do decoder: {DECODER_MODEL_NAME}")
    # Usamos o tokenizer do modelo de linguagem em português
    decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME)
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder_tokenizer.bos_token = decoder_tokenizer.bos_token or decoder_tokenizer.eos_token
    processor.tokenizer = decoder_tokenizer

    model.decoder.resize_token_embeddings(len(processor.tokenizer))


    if processor.tokenizer.pad_token is None:
        logger.warning("Tokenizer não possui pad_token. Usando eos_token como pad_token.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- Configuração Essencial do Modelo ---
    # Define os tokens especiais no config do modelo, que são usados pelo método .generate()
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Configuração do feixe de busca (beam search) para geração
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    if use_peft:
        logger.info("Configurando o modelo com PEFT/LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"], # Módulos de atenção no encoder e decoder
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Modelo envelopado com LoRA. Parâmetros treináveis:")
        model.print_trainable_parameters()
    else:
        logger.info("Treinando o modelo completo (sem PEFT/LoRA).")

    return model, processor
