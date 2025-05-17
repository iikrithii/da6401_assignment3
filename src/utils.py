import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from src.vocab import PAD_TOKEN, EOS_TOKEN

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded, src_lens, tgt_lens


def sequence_accuracy(
    pred_indices: List[List[int]],
    gold_indices: List[List[int]],
    pad_idx: int,
    sos_idx: int,
    eos_idx: int
) -> float:
    """
    Compute exact‐match accuracy, ignoring pad tokens (and anything
    after an EOS) in both prediction and gold.
    """
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