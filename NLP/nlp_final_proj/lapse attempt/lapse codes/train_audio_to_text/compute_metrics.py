# ================================================
# File: compute_metrics.py
# ================================================

import numpy as np
import evaluate

def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_strs = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_strs, references=label_strs)
        return {"wer": wer}

    return compute_metrics
