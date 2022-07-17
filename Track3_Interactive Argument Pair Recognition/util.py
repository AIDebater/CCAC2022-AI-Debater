# Utilities for baseline models.
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import Dict, List

import numpy as np


def str2dict(s: str) -> Dict[str, str]:
    return dict(zip(('cq', 'cr', 'q', '1', '2', '3', '4', '5'),
                    s.replace('\n\n','\n').split('\n')[1:16:2]))


def get_labels(data_path: Path, ids: List[int]) -> np.ndarray:
    return np.array([int((data_path / f'{x}.txt').open(encoding='gb18030').read()[-1])
                     for x in ids])


def compute_acc(label: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(label == np.argmax(pred, axis=1) + 1)


def compute_mrr(label: np.ndarray, pred: np.ndarray) -> float:
    rank: np.ndarray = np.argsort(-pred, axis=1).argsort(axis=1)
    return np.mean(1 / (1 + rank[np.arange(label.shape[0]), label - 1]))
