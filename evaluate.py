import os
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import Seq2SeqDataset
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import collate_fn, load_checkpoint, sequence_accuracy
from src.vocab   import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq model')
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--test_file',   type=str, required=True)
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--output_dir',  type=str, default='predictions')
    # The following parameters must match your model training parameters
    parser.add_argument('--emb_dim',     type=int, required=True)
    parser.add_argument('--hid_dim',     type=int, required=True)
    parser.add_argument('--enc_layers',  type=int, required=True)
    parser.add_argument('--dec_layers',  type=int, required=True)
    parser.add_argument('--cell_type',   type=str, choices=['RNN','GRU','LSTM'], required=True)
    parser.add_argument('--dropout',     type=float, default=0.0)
    parser.add_argument('--beam_width', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load full checkpoint along with vocabulary etc.
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model_state = ckpt['model_state']
    src_vocab   = ckpt['src_vocab']
    tgt_vocab   = ckpt['tgt_vocab']


    encoder = Encoder(
        input_dim=src_vocab.size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        n_layers=args.enc_layers,
        cell_type=args.cell_type,
        dropout=args.dropout
    )
    decoder = Decoder(
        output_dim=tgt_vocab.size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        n_layers=args.dec_layers,
        cell_type=args.cell_type,
        dropout=args.dropout
    )
    model = Seq2Seq(encoder, decoder, args.beam_width, device)
    model.load_state_dict(model_state)
    model.to(device).eval()

    # Preparing test data
    test_ds = Seq2SeqDataset(args.test_file, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    all_preds, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for src_batch, tgt_batch, _, _ in test_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            preds = []
            for src in src_batch:
                src = src.unsqueeze(0)
                if model.beam_width > 1:
                    seq = model.beam_search(
                        src,
                        max_len= tgt_batch.size(1),
                        sos_idx = tgt_vocab.char2idx[SOS_TOKEN],
                        eos_idx = tgt_vocab.char2idx[EOS_TOKEN],
                        beam_width = model.beam_width
                    )
                else:
                    logits = model(src, torch.zeros_like(src), teacher_forcing_ratio=0.0)
                    seq = logits.argmax(-1).squeeze(0).tolist()
                preds.append(seq)
            targets = [seq[1:-1] for seq in tgt_batch.tolist()]
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_inputs.extend(src_batch.cpu().tolist())

    pad_idx = tgt_vocab.char2idx[PAD_TOKEN]
    sos_idx = tgt_vocab.char2idx[SOS_TOKEN]
    eos_idx = tgt_vocab.char2idx[EOS_TOKEN]

    acc = sequence_accuracy(
        all_preds,
        all_targets,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    print(f"Exact-match (cleaned) Accuracy: {acc:.2f}%")

    # Decoding back to strings
    decoded_in = [src_vocab.indices_to_sentence(seq) for seq in all_inputs]
    decoded_tg = [tgt_vocab.indices_to_sentence(seq) for seq in all_targets]
    decoded_pr = [tgt_vocab.indices_to_sentence(seq) for seq in all_preds]

    # Saving the predictions in tsv file
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame({
        'input': decoded_in,
        'target': decoded_tg,
        'pred': decoded_pr
    })
    for col in ['input', 'target', 'pred']:
        df[col] = df[col].str.replace('<pad>', '', regex=False)
        df[col] = df[col].str.replace('<sos>', '', regex=False)
        df[col] = df[col].str.strip()
    df.to_csv(os.path.join(args.output_dir, 'predictions.tsv'),
              sep='\t', index=False)

if __name__ == '__main__':
    main()
