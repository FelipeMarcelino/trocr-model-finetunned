import evaluate

# Carregando as métricas
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):
    """Calcula as métricas WER e CER a partir das predições do modelo.
    """
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decodifica os IDs para texto, ignorando tokens especiais
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Substitui -100 (tokens de padding ignorados no loss) pelo ID de pad do tokenizer
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # Calcula WER e CER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}
