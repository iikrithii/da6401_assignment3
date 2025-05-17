import pandas as pd
import torch
from torch.utils.data import Dataset
from src.vocab import Vocab, SOS_TOKEN, EOS_TOKEN

class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tsv_file: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = 50
    ):

        raw = pd.read_csv(
            tsv_file,
            sep='\t',
            header=None,
            usecols=[0, 1],
            names=['dev', 'lat'],
            dtype=str,
            encoding='utf-8',
            keep_default_na=False,
            na_filter=False
        )

        raw = raw.fillna('').astype(str)

        # Swaping because src is Latin and tgt is Devanagari
        self.data = pd.DataFrame({
            'src': raw['lat'],
            'tgt': raw['dev'],
        })

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        # Building character vocabs
        for src_word, tgt_word in zip(self.data['src'], self.data['tgt']):
            self.src_vocab.add_sentence(src_word)
            self.tgt_vocab.add_sentence(tgt_word)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data.iloc[idx]['src']
        tgt = self.data.iloc[idx]['tgt']

        src = str(src)
        tgt = str(tgt)

        src_idx = self.src_vocab.sentence_to_indices(src)
        tgt_idx = (
            [self.tgt_vocab.char2idx[SOS_TOKEN]] +
            self.tgt_vocab.sentence_to_indices(tgt) +
            [self.tgt_vocab.char2idx[EOS_TOKEN]]
        )

        return (
            torch.tensor(src_idx, dtype=torch.long),
            torch.tensor(tgt_idx, dtype=torch.long)
        )
