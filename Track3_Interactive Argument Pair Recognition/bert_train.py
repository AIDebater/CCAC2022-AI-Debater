# Baseline model based on BERT (training script).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import List

import torch
from transformers import set_seed

from bert import BertForSequenceClassification, BertTokenizer, train

if __name__ == '__main__':
    set_seed(2022)
    data_path = Path('data')
    model_path = Path('model')

    train_ids: List[int] = list(range(1, 251))
    valid_ids: List[int] = list(range(251, 301))

    model_card = 'bert-base-chinese'
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_card)
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        model_card, num_labels=2
    )

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    train(data_path, train_ids, valid_ids, tokenizer, model, model_path)
