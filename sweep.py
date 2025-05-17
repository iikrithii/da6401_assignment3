import pprint
import wandb
from train import main

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'best_val_acc',
        'goal': 'maximize'
    },

    'parameters': {
        'emb_dim': {
            'values': [16, 32, 64, 256]
        },
        'hid_dim': {
            'values': [16, 32, 64, 256]
        },
        'enc_layers': {
            'values': [1, 2, 3]
        },
        'dec_layers': {
            'values': [1, 2, 3]
        },
        'cell_type': {
            'values': ['RNN', 'GRU', 'LSTM']
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'beam_width': {
            'values': [1, 3, 5]
        },
        'optimizer': {
            'values': ['Adam','SGD','RMSprop','NAdam']
        },
        'lr': {
            'values': [0.001, 0.0005, 0.0001]
        },
        'batch_size': {
            'values': [64, 128]
        },
        'epochs': {
            'values': [15]
        }
    }
}

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment3", entity="ns25z040-indian-institute-of-technology-madras")
print("Sweep ID:", sweep_id)

wandb.agent(sweep_id, function=main)