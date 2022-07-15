# Track 1: Argument Extraction and Claim Stance Classification

## 0. Set Up
### 0.1 Dataset
This project uses `train.txt`, `dev.txt` and `test.txt` provided in the first phase of the competition. Please refer to the [official website](http://www.fudan-disc.com/sharedtask/AIDebater22/index.html) for competition registration and dataset downloading.

The dataset used in this project contains 55,544 training instances, 7,057 validation instances and 7,065 test instances. Each line contains three fields: `TOPIC<tab>CANDIDATE_SENT<tab>LABEL`.


### 0.2 Requirements
- Python >= 3.6 and PyTorch >= 1.6
- simpletransformers [link](https://github.com/ThilinaRajapakse/simpletransformers).


### 0.3 Other Settings [Updated]
We train our models for 10 epochs. Batch size is set as 128. We adopt negative sampling strategy and use 5 random negative samples for each claim.


## 1. Training
We use sentence-pair classification model based on _roberta_base_ as our baseline. You can train the baseline model by running 
```
python train.py
```

On each epoch end, the checkpoint will be saved to `outputs/`. The model achieves the best performance on the dev set will be saved to `outputs/best_model/`.

## 2. Evaluation
To evaluate the trained model and generate the submission file, you can simply run
```
python main.py
```
The program will generate the model's evaluation results (_eval_results.txt_) and the submission TXT file (_submission.csv_), which can be found in outputs/.