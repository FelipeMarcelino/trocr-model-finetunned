import logging
import sys
from pathlib import Path


def setup_logger(log_dir: Path):
    """Configura o logger para salvar em arquivo e mostrar no console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    # Formato do log
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Logger principal
    logger = logging.getLogger("trocr")
    logger.setLevel(logging.INFO)
    logger.propagate = False # Evita logs duplicados

    # Handler para o console
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # Handler para o arquivo
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
