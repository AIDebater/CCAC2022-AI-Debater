# Track 4: Argument Generation

## 0. Set Up

### 0.1 Dataset

The dataset used in this project consists of various claims, each related to a number of arguments. All arguments are stored in text files, one argument per line and one file per claim.

Please refer to the [official website](http://www.fudan-disc.com/sharedtask/AIDebater22/index.html) for competition registration and dataset downloading. All data files should be put in `data/`.

### 0.2 Requirements

- jieba
- nltk
- pandas
- pytorch
- rouge
- simplet5
- transformers

### 1. Training

We provide a baseline model fine-tuned on `Langboat/mengzi-t5-base`. You can train the baseline model by running

```{bash}
python train.py
```

The model will only be fine-tuned for one epoch. And then it will be saved in `model/`.

### 2. Evaluation

To evaluate the trained model, you can simply run

```{bash}
python eval.py
```

The program will output predictions of the dev set to `output/`. Each claim in the dev set will have its own output file containing five arguments, one per line. The program will also report performance metrics like BLEU and ROUGE.
