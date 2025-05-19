import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import font_manager
import wandb

wandb.init(project="DA6401_Assignment3", name="confusion_attention")

df = pd.read_csv("predictions_attention/predictions.tsv", sep="\t")

targets = df["target"].astype(str).tolist()
preds = df["pred"].astype(str).tolist()

char_set = set()
for t, p in zip(targets, preds):
    char_set.update(t)
    char_set.update(p)

char_set.add(' ')

char_list = sorted(list(char_set))
char2idx = {char: i for i, char in enumerate(char_list)}

matrix = np.zeros((len(char_list), len(char_list)), dtype=int)

for t, p in zip(targets, preds):
    max_len = max(len(t), len(p))
    t = t.ljust(max_len, ' ')
    p = p.ljust(max_len, ' ')
    for tc, pc in zip(t, p):
        matrix[char2idx[tc], char2idx[pc]] += 1

display_labels = ['<space>' if c == ' ' else c for c in char_list]

font_path = "Nirmala.ttf"
custom_font = font_manager.FontProperties(fname=font_path)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
disp.plot(ax=ax, include_values=False, xticks_rotation=0, cmap='Blues')

ax.set_xticklabels(display_labels, fontproperties=custom_font)
ax.set_yticklabels(display_labels, fontproperties=custom_font)

ax.set_title("Character-wise Confusion Matrix", fontproperties=custom_font)

plt.tight_layout()
wandb.log({ "Confusion Matrix": wandb.Image(fig) })
