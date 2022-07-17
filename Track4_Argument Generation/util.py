# Utilities for baseline models.
#
# Author: Jingcong Liang

from pathlib import Path
from typing import Dict, List, Literal


def read_data(data_path: Path, mode: Literal['train', 'test']) -> Dict[str, List[str]]:
    assert mode in ('train', 'test')
    with (data_path / f'{mode}.txt').open('r', encoding='utf8') as f:
        all_claims: List[str] = [x[:x.index('.txt')]for x in f.readlines()]

    data: Dict[str, List[str]] = {}

    for claim in all_claims:
        data_file: Path = data_path / f'{claim}.txt'
        title: str = f'主题：{claim}'
        data[title] = []

        if data_file.exists():
            with data_file.open('r', encoding='utf8') as f:
                for argument in f.readlines():
                    argument = argument.strip()

                    while argument != '':
                        data[title].append(argument[:128])
                        argument = argument[128:]

    return data
