import torch
import torch.nn as nn
from typing import Optional
import heapq

#Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int,
                 n_layers: int, cell_type: str = 'LSTM', dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)

    def forward(self, src, src_len=None):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

#Decoder model
class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int,
                 n_layers: int, cell_type: str = 'LSTM', dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

#Combining Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, beam_width, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.beam_width = beam_width

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        batch_size = src.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden = self.encoder(src)

        # First input to decoder is <sos>
        input = trg[:, 0]
        for t in range(1, max_len):
            pred, hidden = self.decoder(input, hidden)
            outputs[:, t] = pred
            top1 = pred.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs
    
    def beam_search(self, src, max_len, sos_idx, eos_idx, beam_width):
        """
        src: [batch=1, src_len]
        Returns: best output sequence of token indices
        """
        enc_outputs, hidden = self.encoder(src)
        beams = [(0.0, [sos_idx], hidden)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, seq, hid in beams:
                last_token = seq[-1]
                if last_token == eos_idx:
                    completed.append((score, seq))
                    continue
                logits, next_hid = self.decoder(
                    torch.LongTensor([last_token]).to(self.device),
                    hid
                )
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                # pick top-k continuations for beam search
                topk_logps, topk_idxs = log_probs.topk(beam_width)
                for logp, idx in zip(topk_logps.tolist(), topk_idxs.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + logp
                    new_beams.append((new_score, new_seq, next_hid))
            # keep only top k overall
            beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])
            if not beams:
                break
        completed.extend(beams)
        # pick the final sequence which has the highest score
        best_seq = max(completed, key=lambda x: x[0])[1]
        return best_seq