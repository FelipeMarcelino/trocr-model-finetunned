import logging

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor

from trocr.config import DECODER_MODEL_NAME, ENCODER_MODEL_NAME, LORA_ALPHA, LORA_DROPOUT, LORA_R, MAX_TARGET_LENGTH

logger = logging.getLogger("trocr.model")

def initialize_model(use_peft: bool = False):
    """Constrói um modelo VisionEncoderDecoder combinando um encoder de visão pré-treinado
    com um decoder de linguagem em português, e opcionalmente aplica PEFT/LoRA.
    """
    # 1. Carrega os componentes individuais
    logger.info(f"Carregando ENCODER de imagem de: {ENCODER_MODEL_NAME}")
    image_processor = ViTImageProcessor.from_pretrained(ENCODER_MODEL_NAME)

    logger.info(f"Carregando DECODER de texto (tokenizer) de: {DECODER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME)

    # O TrOCRProcessor é um wrapper conveniente para os dois
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 2. Constrói o modelo a partir dos componentes pré-treinados
    logger.info("Construindo o modelo VisionEncoderDecoder a partir do encoder e decoder...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=ENCODER_MODEL_NAME,
        decoder_pretrained_model_name_or_path=DECODER_MODEL_NAME,
    )

    # 3. Configura os tokens especiais (ESSENCIAL)
    logger.info("Configurando os tokens especiais para o novo decoder...")
    # O GPT2 não tem alguns tokens, então definimos com base no que ele tem
    tokenizer.pad_token = tokenizer.eos_token

    # Sincroniza a configuração do modelo com o tokenizador
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = len(tokenizer)
    model.config.decoder.vocab_size = len(tokenizer)

    # 4. Configura os parâmetros de geração explicitamente
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.num_beams = 4
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0

    if use_peft:
        logger.info("Configurando o modelo com PEFT/LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["c_attn"],
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
