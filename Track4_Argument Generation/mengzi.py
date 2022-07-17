# Baseline model based on Mengzi.
#
# Author: Jingcong Liang

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from simplet5 import SimpleT5
from transformers import T5ForConditionalGeneration, T5Tokenizer

from util import read_data


class MengziSimpleT5(SimpleT5):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

    def from_pretrained(self, model_path: Optional[Path] = None) -> None:
        source: Union[Path, str] = 'Langboat/mengzi-t5-base' if model_path is None else model_path
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(source)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(source)
        self.model = self.model.to(self.device)

    def save_pretrained(self, model_path: Path) -> None:
        self.tokenizer.save_pretrained(model_path)
        self.model.save_pretrained(model_path)


def make_dataset(data_path: Path, mode: Literal['train', 'test']) -> pd.DataFrame:
    assert mode in ('train', 'test')
    data: List[Tuple[str, str]] = []

    for claim, arguments in read_data(data_path, mode).items():
        for i in range(len(arguments)):
            argument = ''

            for j in range(i, len(arguments)):
                argument = argument + arguments[j]
                if len(argument) > 128:
                    break

            data.append((claim, argument))

    return pd.DataFrame(data, columns=('source_text', 'target_text'))
