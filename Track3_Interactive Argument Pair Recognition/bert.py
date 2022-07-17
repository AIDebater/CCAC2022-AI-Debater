# Baseline model based on BERT (core module).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import choice
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BatchEncoding, BertForSequenceClassification, BertTokenizer

from util import compute_acc, compute_mrr, get_labels, str2dict


@dataclass
class Track3Dataset(Dataset):
    data_path: Path
    ids: List[int]
    mode: Literal['train', 'test']
    tokenizer: BertTokenizer

    _TAGS: ClassVar = '12345'

    def __post_init__(self) -> None:
        assert self.mode in ('train', 'test')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, Optional[List[int]]]:
        s: str = (self.data_path / f'{self.ids[idx]}.txt').open(encoding='gb18030').read()
        record: Dict[str, str] = str2dict(s)
        answer: Optional[str] = s[-1] if '互动论点：' in s else None

        first_text: List[str]
        second_text: List[str]
        label: Optional[List[int]]

        if self.mode == 'train':
            first_text = [record['q'], record['q']]
            second_text = [record[answer], record[choice(self._TAGS.replace(answer, ''))]]
            label = [1, 0]
        else:
            first_text = [record['q'] for _ in self._TAGS]
            second_text = [record[tag] for tag in self._TAGS]
            label = [int(tag == answer) for tag in self._TAGS] if answer is not None else None

        return self.tokenizer(first_text, second_text, truncation=True), label


def create_mini_batch(
    samples: List[Tuple[BatchEncoding, Optional[List[int]]]]
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    all_tokens: DefaultDict[str, List[List[int]]] = defaultdict(list)
    all_labels: List[int] = []

    for tokens, labels in samples:
        for k, v in tokens.items():
            all_tokens[k].extend(v)
        if labels is not None:
            all_labels.extend(labels)

    return {k: pad_sequence([torch.tensor(x) for x in v], batch_first=True) for k, v in
            all_tokens.items()}, torch.tensor(all_labels) if len(all_labels) > 0 else None


def inputs_to_device(inputs: Dict[str, torch.Tensor],
                     device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in inputs.items()}


def get_predictions(data_path: Path, ids: List[int], tokenizer: BertTokenizer,
                    model: BertForSequenceClassification) -> np.ndarray:
    dataset = Track3Dataset(data_path, ids, 'test', tokenizer)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=create_mini_batch)
    pred: List[np.ndarray] = []

    with torch.no_grad():
        model.eval()

        for inputs, _ in loader:
            inputs: Dict[str, torch.Tensor] = inputs_to_device(inputs, model.device)
            outputs: torch.Tensor = model(**inputs)
            pred.append(outputs[0][:, 1].cpu().numpy())

    return np.stack(pred)


def eval_model(data_path: Path, ids: List[int], tokenizer: BertTokenizer,
               model: BertForSequenceClassification,
               return_mrr: bool = False) -> Union[float, Tuple[float, float]]:
    label: np.ndarray = get_labels(data_path, ids)
    pred: np.ndarray = get_predictions(data_path, ids, tokenizer, model)
    acc: float = compute_acc(label, pred)

    if return_mrr:
        mrr: float = compute_mrr(label, pred)
        return acc, mrr
    else:
        return acc


def train(data_path: Path, train_ids: List[int], valid_ids: List[int], tokenizer: BertTokenizer,
          model: BertForSequenceClassification, model_path: Path) -> None:
    dataset = Track3Dataset(data_path, train_ids, 'train', tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=create_mini_batch)
    optimizer = Adam(model.parameters(), lr=2e-6)

    best_acc: float = eval_model(data_path, valid_ids, tokenizer, model)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    for epoch in range(20):
        model.train()
        running_loss: float = 0.0

        for inputs, labels in tqdm(loader):
            inputs: Dict[str, torch.Tensor] = inputs_to_device(inputs, model.device)
            labels: torch.Tensor = labels.to(model.device)

            optimizer.zero_grad()
            loss: torch.Tensor = model(**inputs, labels=labels)[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc: float = eval_model(data_path, valid_ids, tokenizer, model)
        print(f'[epoch {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f} best: {best_acc:.3f}')

        if acc >= best_acc:
            best_acc = acc
            model.save_pretrained(model_path)
            print('current best model saved')

    print(f'best acc: {best_acc:.3f}')
