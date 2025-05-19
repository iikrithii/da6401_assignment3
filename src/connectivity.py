import numpy as np
import plotly.graph_objects as go
import torch

# your 20‐step colour palette
PALETTE = [
  "#85c2e1", "#89c4e2", "#95cae5", "#99cce6", "#a1d0e8",
  "#b2d9ec", "#baddee", "#c2e1f0", "#eff7fb", "#f9e8e8",
  "#f9e8e8", "#f9d4d4", "#f9bdbd", "#f8a8a8", "#f68f8f",
  "#f47676", "#f45f5f", "#f34343", "#f33b3b", "#f42e2e"
]

def get_clr(val: float) -> str:
    idx = int(np.clip(val*19, 0, 19))
    return PALETTE[idx]

def plot_text_attention(inp_str, out_str, attn, title="Attention"):


    T_real = len(out_str)
    S_real = len(inp_str)

    attn = attn[:T_real, :(S_real)]

    vmin, vmax = attn.min(), attn.max()
    attn = (attn - vmin) / (vmax - vmin + 1e-8)
    T, S = attn.shape

    fig = go.Figure()

    # 1) Dummy heatmap to create a horizontal colorbar on top
    legend_trace = go.Heatmap(
        z=[np.linspace(0,1,20)],
        colorscale=[[i/19, get_clr(i/19)] for i in range(20)],
        showscale=True,
        colorbar=dict(
            title="Attention",
            orientation="h",    # horizontal
            thickness=15,
            len=0.7,
            x=0.5,              # center above
            y=1.15,             # above the top of plot
            xanchor="center",
            yanchor="bottom",
            tickmode="array",
            tickvals=[0,0.5,1],
            ticktext=["0%","50%","100%"]
        ),
        opacity=0.0
    )
    fig.add_trace(legend_trace)

    # 2) Build shapes & annotations per time‐step
    all_shapes = []
    all_annots = []
    for t in range(T):
        shapes = []
        annots = []
        for i, ch in enumerate(inp_str):
            c = get_clr(attn[t,i])
            shapes.append(dict(
                type="rect", x0=i, x1=i+1, y0=0, y1=1,
                fillcolor=c, line_width=0
            ))
            annots.append(dict(
                x=i+0.5, y=0.5, text=ch,
                showarrow=False, font=dict(size=20, color="black")
            ))
        all_shapes.append(shapes)
        all_annots.append(annots)

    # 3) Slider steps
    steps = []
    for t, o_ch in enumerate(out_str):
        steps.append(dict(
            method="relayout",
            args=[{
                "shapes": all_shapes[t],
                "annotations": all_annots[t]
            }],
            label=o_ch
        ))

    # 4) Initial layout: step 0
    fig.update_layout(
        title=dict(
            text=f"{title} – Predicted: <b>{out_str}</b>",
            y=0.9,  # nudge down so colorbar fits
        ),
        xaxis=dict(showticklabels=False, range=[0, S]),
        yaxis=dict(showticklabels=False, range=[0, 1]),
        shapes=all_shapes[0],
        annotations=all_annots[0],
        sliders=[dict(
            active=0,
            pad={"t": 80},  # push slider down a bit
            currentvalue={"prefix": "Decoding char: "},
            steps=steps
        )],
        margin=dict(l=20, r=20, t=120, b=80),
        height=350,
    )

    return fig

# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    inp  = "ghar"
    out  = "घर"
    attn = np.random.rand(len(out), len(inp))
    attn = attn / attn.sum(axis=1, keepdims=True)

    fig = plot_text_attention(inp, out, attn, title="Dev–Lat Transliteration")
    fig.show()

    # to log in W&B:
    import wandb
    wandb.init(project="dakshina‐attn", name="figly")
    wandb.log({"attention_map": fig})
    wandb.finish()
