import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from src.vocab import PAD_TOKEN, EOS_TOKEN
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded, src_lens, tgt_lens

#For calculating exact word match accuracy
def sequence_accuracy(
    pred_indices: List[List[int]],
    gold_indices: List[List[int]],
    pad_idx: int,
    sos_idx: int,
    eos_idx: int
) -> float:
    def clean(seq: List[int]) -> List[int]:
        out = []
        for idx in seq:
            if idx == eos_idx:
                break
            if idx != pad_idx and idx != sos_idx:
                out.append(idx)
        return out

    correct = 0
    for p_seq, g_seq in zip(pred_indices, gold_indices):
        if clean(p_seq) == clean(g_seq):
            correct += 1

    return 100.0 * correct / len(gold_indices)


def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)

def load_checkpoint(filename: str, device: torch.device) -> dict:
    return torch.load(filename, map_location=device)

# For plotting heatmaps
hindi_font = FontProperties(fname = "Nirmala.ttf")
def plot_attention(attention_weights, input_tokens, output_tokens, title, ax):
    T_real = len(output_tokens)
    S_real = len(input_tokens)
    attention_weights = attention_weights[:(T_real), 1:(S_real+1)]
    cax = ax.matshow(attention_weights, cmap='viridis')
    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    ax.set_xticklabels(input_tokens, fontsize=10)
    ax.set_yticklabels(output_tokens, fontsize=10, fontproperties=hindi_font)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title, fontsize=12, fontproperties=hindi_font)
    ax.tick_params(axis='x', rotation=0)

def plot_attention_grid(attention_list, input_list, output_list, titles):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.6, hspace=0.8)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(attention_list):
                plot_attention(attention_list[idx], input_list[idx], output_list[idx], titles[idx], axes[i, j])
    return fig