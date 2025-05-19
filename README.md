# DA6401 Assignment 3: Transliteration system using RNNs

## Introduction

This repository contains the code for **Assignment 3** of DA6401 (Introduction to Deep Learning). The four main objectives of the assignment are:

1. Build a sequence‐to‐sequence model with Recurrent Neural Networks.  
2. Compare different RNN cell types: vanilla RNN, LSTM, and GRU.  
3. Implement and understand attention networks and see how they overcome the limitations of vanilla seq2seq models.  
4. Visualize these attentions using interactive plots.


---

## Problem Statement

The objective of this assignment is to build a character-level sequence-to-sequence (seq2seq) model for transliteration using the Dakshina dataset released by Google. Each data sample consists of a pair: a word written in the Latin script (e.g., "ghar") and its corresponding representation in the native Devanagari script (e.g., "घर"). The goal is to learn a mapping function that can take a romanized input string and accurately generate its transliterated form in the native script.  I have chosen Hindi for this assignment as the native language.

---

## Repository Structure

```
├── dakshina_dataset_v1.0/       # Dataset
├── predictions_attention/       # Test‑set predictions with attention
│   └── predictions.tsv
├── predictions_vanilla/         # Test‑set predictions without attention
│   └── predictions.tsv
├── src/                         # Core code
│   ├── connectivity.py          # Connectivity visualization code for Question 6
│   ├── dataset.py               # Data loading & preprocessing
│   ├── model.py                 # Seq2seq and attention model
│   ├── utils.py                 # Helper functions
│   ├── plot_confusion.py        # Plot confusion matrix 
│   └── vocab.py                 # Build source/target vocabularies
├── evaluate.py                  # Evaluate trained model
├── requirements.txt             # Python dependencies
├── sweep.py                     # wandb sweep configuration (vanilla RNN)
├── sweep2.py                    # wandb sweep configuration (attention)
└── train.py                     # Main training script

```


---

## Hyperparameters & CLI Arguments

Below is the summary for all the arguments in the `train.py` file

| Parameter         | CLI Argument                   | Description                                                          | Default                                         |
|-------------------|------------------------|----------------------------------------------------------------------|-------------------------------------------------|
| Training file     | `--train_file`         | Path to the training TSV                                             | `path-to-train-dataset` |
| Validation file   | `--val_file`           | Path to the validation TSV                                       | `path-to-validation-dataset`   |
| wandb Entity      | `--wandb_entity`       | Your Wandb account name                                           | `your-entity-here`                              |
| Batch size        | `--batch_size`         | Number of examples per batch                                         | `64`                                            |
| Embedding dim     | `--emb_dim`            | Character embedding size                                              | `128`                                           |
| Hidden dim        | `--hid_dim`            | Hidden state size for encoder & decoder                              | `256`                                           |
| Encoder layers    | `--enc_layers`         | Number of stacked RNN layers in the encoder                          | `1`                                             |
| Decoder layers    | `--dec_layers`         | Number of stacked RNN layers in the decoder                          | `1`                                             |
| Cell type         | `--cell_type`          | RNN cell: `RNN`, `GRU`, or `LSTM`                                    | `LSTM`                                          |
| Dropout rate      | `--dropout`            | Dropout probability                      | `0.1`                                           |
| Learning rate     | `--lr`                 | Learning rate                                                | `1e-3`                                          |
| Epochs            | `--epochs`             | Number of training epochs                                            | `10`                                            |
| Save path         | `--save_path`          | Where to write the best model checkpoint                             | `models/model.pt`                               |
| wandb Project     | `--project`            | W&B project name                                                     | `DA6401_Assignment3`                            |
| Optimizer         | `--optimizer`          | Optimizer: `Adam`, `SGD`, `RMSprop`, `NAdam`, or `AdamW`             | `Adam`                                          |
| Beam width        | `--beam_width`         | Beam size during decoding with beam search                                           | `1`                                             |
| Attention flag    | `--attention`          | Enable attention mechanism                                          | `False`                                         |

---

## Guidelines for Implementing code

1. Download the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina) and add it to repository.

2. Run `requirements.txt` to install required packages for training and evaluating the models.

### Train & Evaluate **Vanilla** Model

You can run the following prompts in command line argumenent for training and evaluating the best vanilla model.

```bash
python train.py --train_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv --val_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv --batch_size 256 --emb_dim 512 --hid_dim 512 --enc_layers 4 --dec_layers 3 --cell_type GRU --dropout 0.3 --lr 0.0005 --epochs 25 --beam_width 1 --optimizer Adam --save_path models/best_model_vanilla.pt --project your-project-name --wandb_entity your-entity 
```

```bash
python evaluate.py --model_path models/best_model_vanilla.pt --test_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv --batch_size 256 --output_dir predictions_vanilla --emb_dim 512 --hid_dim 512 --enc_layers 4 --dec_layers 3 --cell_type GRU --dropout 0.3 --beam_width 1
```

### Train & Evaluate **Attention** Model

You can run the following prompts in command line argumenent for training and evaluating the best attention model.

```bash
python train.py --train_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv --val_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv --batch_size 256 --emb_dim 512 --hid_dim 512 --enc_layers 3 --dec_layers 4 --cell_type GRU --dropout 0.3 --lr 0.0005 --epochs 25 --beam_width 1 --optimizer Adam --save_path models/best_model_attention.pt --project your-project-name --wandb_entity your-entity --attention True
```

```bash
python evaluate.py --model_path models/best_model_attention.pt --test_file dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv --batch_size 256 --output_dir predictions_attention --emb_dim 512 --hid_dim 512 --enc_layers 3 --dec_layers 4 --cell_type GRU --dropout 0.3 --beam_width 1
```

## Wandb Report

The Wandb report for this assignment can be found here:
[Report](https://wandb.ai/ns25z040-indian-institute-of-technology-madras/DA6401_Assignment3/reports/DA6401-Assignment-3--VmlldzoxMjgzMjEyNQ?accessToken=zyd22tbh7iumhfcgcs5stko5zv594bhu8tafno6rxvi6yetqwiw4lqknziqtrmp9)