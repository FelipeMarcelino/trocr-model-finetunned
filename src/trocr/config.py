from pathlib import Path

# --- Diretórios ---
# Usamos Path para compatibilidade entre sistemas operacionais
ROOT_DIR = Path(__file__).parent.parent.parent
PROJECT_DIR = ROOT_DIR / "trocr"
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "labels.csv"

# Diretórios de saída
OUTPUT_DIR = ROOT_DIR
LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard_runs"

# --- Configurações do Modelo ---
# Encoder: Modelo de visão pré-treinado da Microsoft
ENCODER_MODEL_NAME = "google/vit-base-patch16-224"
# Decoder: Modelo de linguagem em português. O tokenizer dele será usado.
DECODER_MODEL_NAME = "pierreguillou/gpt2-small-portuguese"
# Processor: Usamos o de um modelo TrOCR completo para obter o processador de imagem correto
PROCESSOR_MODEL_NAME = "microsoft/trocr-base-handwritten"


# --- Configurações de Treinamento ---
DEVICE = "cuda" # "cuda" ou "cpu"
MAX_TARGET_LENGTH = 128 # Comprimento máximo das sequências de texto
TRAIN_TEST_SPLIT_RATIO = 0.1 # 10% dos dados para teste
RANDOM_STATE = 42

# --- Configurações de PEFT/LoRA (se usado) ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
