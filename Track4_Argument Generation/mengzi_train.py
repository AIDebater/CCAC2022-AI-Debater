# Baseline model based on Mengzi.
#
# Author: Jingcong Liang

from pathlib import Path

import pandas as pd
import torch
from transformers import set_seed

from mengzi import make_dataset, MengziSimpleT5


if __name__ == '__main__':
    set_seed(2022)
    data_path = Path('data')
    model_path = Path('model')

    train_set: pd.DataFrame = make_dataset(data_path, 'train')
    valid_set: pd.DataFrame = make_dataset(data_path, 'test')

    model = MengziSimpleT5(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.from_pretrained()

    model.train(train_set, valid_set, source_max_token_len=32, target_max_token_len=256,
                max_epochs=1, outputdir='log', dataloader_num_workers=32)
    model.save_pretrained(model_path)
