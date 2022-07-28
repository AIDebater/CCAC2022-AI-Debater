# Track 3: Interactive Argument Pair Recognition

## 0. Set Up

### 0.1 Dataset

The dataset used in this project consists of text-file-based instances. Each instance file follows the following format, where `K` denotes the index of the correct argument (indexed from $1$), and other placeholders argumentative texts:

```{plain}
辩稿A：
XXXXXXXXXXXX

辩稿B：
YYYYYYYYYYYY

目标论点：
ZZZZZZ

候选论点1：
AAAAAA

候选论点2：
BBBBBB

候选论点3：
CCCCCC

候选论点4：
DDDDDD

候选论点5：
EEEEEE

互动论点：K
```

Please refer to the [official website](http://www.fudan-disc.com/sharedtask/AIDebater22/index.html) for competition registration and dataset downloading. All data files should be put in `data/`.

### 0.2 Requirements

- jieba
- pytorch
- transformers

## 1. Matching-based model

A baseline model based on word matching is provided in `matching.py`, which ranks candidate arguments according to the number of common words with the target arguments. Running `python matching.py` will output predicted ranks of the dev set to `output/matching_result.txt`, meanwhile compute and report relevant performance metrics like accuracy and mean reciprocal rank (MRR). The rank of arguments within each instance are separated by commas (,), consuming only one line.

## 2. BERT-based model

Besides the matching-based model, we provide another baseline based on `bert_base_chinese`.

### 2.1 Training

You can train the BERT-based baseline model by running

```{bash}
python bert_train.py
```

On each epoch end, the current model will be evaluated on the dev set, and the best model will be saved in `model/`.

### 2.2 Evaluation

To evaluate the trained BERT-based model, you can simply run

```{bash}
python bert_eval.py
```

Similar to `matching.py`, the program will output predictions of the dev set to `output/bert_result.txt` and report all performance metrics mentioned above. The output file has the same format as `output/matching_result.txt`, where predictions are expressed as ranks.
