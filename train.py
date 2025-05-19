import os
import argparse

import torch
from torch.utils.data import DataLoader
import wandb

from src.vocab import Vocab, PAD_TOKEN
from src.dataset import Seq2SeqDataset
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import collate_fn, save_checkpoint, sequence_accuracy
from src.vocab   import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    """
    preds = logits.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()

def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2Seq model')
    parser.add_argument('--train_file', type=str, default="dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    parser.add_argument('--val_file',   type=str, default="dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")
    parser.add_argument('--wandb_entity', type=str, default="your-entity-here")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--emb_dim',    type=int, default=128)
    parser.add_argument('--hid_dim',    type=int, default=256)
    parser.add_argument('--enc_layers', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--cell_type',  type=str, choices=['RNN','GRU','LSTM'], default='LSTM')
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--save_path',  type=str, default='models/model.pt')
    parser.add_argument('--project',    type=str, default='DA6401_Assignment3')
    parser.add_argument('--optimizer',  type=str, choices=['Adam','SGD','RMSprop','NAdam','AdamW'], default='Adam')
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--attention', type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project=args.project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config
    print("attention", args.attention)

    run_name = (
        f"bs_{config.batch_size}_"
        f"epochs_{config.epochs}_"
        f"cell_type_{config.cell_type}_"
        f"enc_layers_{config.enc_layers}_"
        f"dec_layers_{config.dec_layers}_"
        f"emb_{config.emb_dim}_"
        f"hid_{config.hid_dim}_"
        f"drop_{config.dropout}_"
        f"beam_{config.beam_width}_"
        f"lr_{config.lr}_"
        f"optim_{config.optimizer}"
    )
    wandb.run.name=run_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Building Vocabularies
    src_vocab = Vocab()
    tgt_vocab = Vocab()
    train_ds = Seq2SeqDataset(args.train_file, src_vocab, tgt_vocab)
    val_ds   = Seq2SeqDataset(args.val_file,   src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # Setting up the model
    enc = Encoder(src_vocab.size, config.emb_dim, config.hid_dim,
                  config.enc_layers, config.cell_type, config.dropout)
    dec = Decoder(tgt_vocab.size, config.emb_dim, config.hid_dim,
                  config.enc_layers, config.cell_type, config.dropout,
                  use_attention=config.attention)
    model = Seq2Seq(enc, dec, args.beam_width, device).to(device)

    optim_map = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'AdamW': torch.optim.AdamW,
        'NAdam': torch.optim.NAdam,
    }
    optimizer = optim_map[config.optimizer](model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.char2idx[PAD_TOKEN])

    best_val_acc = 0.0
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_tgts = [], []
        for src_batch, tgt_batch, _, _ in train_loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            optimizer.zero_grad()
            res = model(src_batch,
                        tgt_batch,
                        teacher_forcing_ratio=0.5,
                        return_attn=True)
            # unpacking it in case of returning the attention weights
            if isinstance(res, tuple):
                outputs, _ = res
            else:
                outputs = res
            out_dim = outputs.size(-1)
            loss = criterion(
                outputs[:,1:].reshape(-1, out_dim),
                tgt_batch[:,1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_preds = outputs.argmax(dim=-1).cpu().tolist()
            batch_tgts  = tgt_batch.cpu().tolist()
            sos_idx = tgt_vocab.char2idx[SOS_TOKEN]
            eos_idx = tgt_vocab.char2idx[EOS_TOKEN]
            pad_idx = tgt_vocab.char2idx[PAD_TOKEN]
            for p, g in zip(batch_preds, batch_tgts):
                train_preds.append(p)
                train_tgts.append(g)

        train_loss /= len(train_loader)
        train_acc = sequence_accuracy(
            train_preds, train_tgts,
            pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
        )

        # Validation on the model for choosing best epoch
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for src_batch, tgt_batch, _, _ in val_loader:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

                res = model(src_batch,
                            tgt_batch,
                            teacher_forcing_ratio=0.0,
                            return_attn=True)
                if isinstance(res, tuple):
                    outputs, _ = res
                else:
                    outputs = res

                out_dim = outputs.size(-1)
                loss = criterion(
                    outputs[:,1:].reshape(-1, out_dim),
                    tgt_batch[:,1:].reshape(-1)
                )
                val_loss += loss.item()

                if model.beam_width > 1:
                    for i in range(src_batch.size(0)):
                        single_src = src_batch[i:i+1]
                        max_len    = tgt_batch.size(1)
                        seq = model.beam_search(
                            single_src,
                            max_len = max_len,
                            sos_idx = tgt_vocab.char2idx[SOS_TOKEN],
                            eos_idx = tgt_vocab.char2idx[EOS_TOKEN],
                            beam_width = model.beam_width
                        )
                        all_preds.append(seq[1:])
                        all_targets.append(tgt_batch[i].tolist()[1:-1])
                else:
                    batch_preds = outputs.argmax(-1).cpu().tolist()
                    all_preds.extend(batch_preds)
                    all_targets.extend([seq[1:-1] for seq in tgt_batch.cpu().tolist()])

        val_loss /= len(val_loader)
        val_acc = sequence_accuracy(
            all_preds, all_targets,
            pad_idx=tgt_vocab.char2idx[PAD_TOKEN],
            sos_idx=tgt_vocab.char2idx[SOS_TOKEN],
            eos_idx=tgt_vocab.char2idx[EOS_TOKEN]
        )

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Logging the metrics to Wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # To save the best epoch model, comment for doing sweep
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
            save_checkpoint({
                'model_state': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, args.save_path)
            # print(f"-> New best model saved (Epoch {epoch}, Val Acc {val_acc:.2f}%)")
    wandb.log({'best_val_acc': best_val_acc})

if __name__ == '__main__':
    main()