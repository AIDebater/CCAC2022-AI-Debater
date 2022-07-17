# Baseline model based on word matching.
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from jieba import lcut

from util import compute_acc, compute_mrr, get_labels, str2dict


def get_predictions(data_path: Path, ids: List[int]) -> np.ndarray:
    res: List[List[int]] = []

    for x in ids:
        record: Dict[str, str] = str2dict((data_path / f'{x}.txt').open(encoding='gb18030').read())
        q_set: Set[str] = set(lcut(record['q']))
        res.append([len(q_set.intersection(lcut(record[str(i)]))) for i in range(1, 6)])

    return np.array(res)


if __name__ == '__main__':
    data_path = Path('data')
    valid_ids: List[int] = list(range(251, 301))

    label: np.ndarray = get_labels(data_path, valid_ids)
    pred: np.ndarray = get_predictions(data_path, valid_ids)

    acc: float = compute_acc(label, pred)
    print(f'acc: {acc:.3f}')
    mrr: float = compute_mrr(label, pred)
    print(f'mrr: {acc:.3f}')
