# Baseline model based on BERT (evaluation script).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import List

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from bert import eval_model


if __name__ == '__main__':
    data_path = Path('data')
    model_path = Path('model')
    valid_ids: List[int] = list(range(251, 301))

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        model_path
    )

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    acc: float
    mrr: float
    acc, mrr = eval_model(data_path, valid_ids, tokenizer, model, return_mrr=True)
    print(f'acc: {acc:.3f}')
    print(f'mrr: {mrr:.3f}')
