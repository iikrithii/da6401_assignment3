PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

class Vocab:
    def __init__(self):
        self.char2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.freqs = {}

    def add_sentence(self, sentence: str):
        for ch in sentence:
            self._add_char(ch)

    def _add_char(self, ch: str):
        if ch not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[ch] = idx
            self.idx2char[idx] = ch
            self.freqs[ch] = 1
        else:
            self.freqs[ch] = self.freqs.get(ch, 0) + 1

    def sentence_to_indices(self, sentence: str) -> list[int]:
        return [self.char2idx.get(ch, self.char2idx[UNK_TOKEN]) for ch in sentence]

    def indices_to_sentence(self, indices: list[int]) -> str:
        chars = []
        for idx in indices:
            ch = self.idx2char.get(idx, UNK_TOKEN)
            if ch == EOS_TOKEN:
                break
            chars.append(ch)
        return ''.join(chars)

    @property
    def size(self) -> int:
        return len(self.char2idx)