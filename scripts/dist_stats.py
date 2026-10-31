"""FP32 weight / input-activation distribution of every nn.Linear, for spotting outliers.

Generic use (any model, e.g. the internal one):
    from scripts.dist_stats import collect_linear_stats
    collect_linear_stats(model, run_fn, out_dir="dist_stats")

run_fn: zero-argument callable that runs ``model`` forward on a handful of inputs with the
same preprocessing as inference. Run it on the FP32 model (no quantizer enabled), otherwise
the statistics describe already-quantized activations.

Outputs in out_dir:
    linear_stats.csv           one row per (layer, weight|act)
    summary.png                per-layer absmax/std of weights and activations
    layer_<i>_<name>.png       for the layers with the strongest activation channel outliers:
                               per-channel absmax, activation histogram, weight histogram

Conformer (this repo):
    python scripts/dist_stats.py
"""
from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn as nn

SAMPLES_PER_CALL = 20000
MAX_SAMPLES = 2_000_000


def _subsample(x, k):
    x = x.flatten()
    if x.numel() <= k:
        return x.clone()
    return x[torch.randint(x.numel(), (k,))]


def _summary(sample, absmax):
    s = sample.double()
    mean, std = s.mean().item(), s.std().item()
    c = s - mean
    kurt = ((c ** 4).mean() / (c ** 2).mean() ** 2).item() if std > 0 else math.nan
    p999 = torch.quantile(s.abs()[:MAX_SAMPLES].float(), 0.999).item()
    return {
        "absmax": absmax,
        "std": std,
        "p99.9": p999,
        "absmax/std": absmax / std if std > 0 else math.nan,
        "absmax/p99.9": absmax / p999 if p999 > 0 else math.nan,
        "kurtosis": kurt,
    }


def _channel_ratio(ch_absmax):
    """max / median of per-channel absmax, plus the top-3 channel indices."""
    med = ch_absmax.median().item()
    ratio = ch_absmax.max().item() / med if med > 0 else math.inf
    top = torch.topk(ch_absmax, min(3, ch_absmax.numel())).indices.tolist()
    return ratio, top


class _ActStat:
    def __init__(self, channels):
        self.ch_absmax = torch.zeros(channels)
        self.absmax = 0.0
        self.samples = []
        self.frame_absmax = []

    def update(self, x):
        x = x.detach().float().reshape(-1, x.shape[-1]).cpu()
        a = x.abs()
        self.ch_absmax = torch.maximum(self.ch_absmax, a.amax(0))
        self.absmax = max(self.absmax, a.max().item())
        self.frame_absmax.append(_subsample(a.amax(1), SAMPLES_PER_CALL))
        self.samples.append(_subsample(x, SAMPLES_PER_CALL))


def collect_linear_stats(model, run_fn, out_dir="dist_stats", plot_top=6):
    os.makedirs(out_dir, exist_ok=True)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for n, m in linears:
        for q in ("input_quantizer", "weight_quantizer"):
            if hasattr(m, q) and not getattr(m, q)._disabled:
                print(f"WARNING: {n}.{q} is enabled; stats will not be FP32")

    stats = {n: _ActStat(m.in_features) for n, m in linears}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, n=n: stats[n].update(inp[0]))
             for n, m in linears]
    model.eval()
    try:
        with torch.no_grad():
            run_fn()
    finally:
        for h in hooks:
            h.remove()

    rows, per_layer = [], []
    for n, m in linears:
        st = stats[n]
        if not st.samples:
            print(f"WARNING: {n} was not called by run_fn")
            continue
        act_sample = torch.cat(st.samples)[:MAX_SAMPLES]
        act = _summary(act_sample, st.absmax)
        act["ch_ratio"], act["top_ch"] = _channel_ratio(st.ch_absmax)
        fa = torch.cat(st.frame_absmax)
        act["frame_ratio"] = fa.max().item() / fa.median().item() if fa.median() > 0 else math.inf

        w = m.weight.detach().float().cpu()
        wt = _summary(_subsample(w, MAX_SAMPLES), w.abs().max().item())
        wt["ch_ratio"], wt["top_ch"] = _channel_ratio(w.abs().amax(0))      # per input channel
        wt["out_ch_ratio"], _ = _channel_ratio(w.abs().amax(1))             # per output channel

        rows.append({"layer": n, "kind": "act", **act})
        rows.append({"layer": n, "kind": "weight", **wt})
        per_layer.append((n, act, wt, st.ch_absmax, act_sample, w))

    fields = ["layer", "kind", "absmax", "std", "p99.9", "absmax/std", "absmax/p99.9",
              "kurtosis", "ch_ratio", "top_ch", "frame_ratio", "out_ch_ratio"]
    with open(os.path.join(out_dir, "linear_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    ranked = sorted(per_layer, key=lambda t: t[1]["ch_ratio"], reverse=True)
    print(f"\n{'layer':<55} {'act absmax/std':>14} {'act ch_ratio':>12} {'act kurt':>9} "
          f"{'w absmax/std':>12} {'w kurt':>7}")
    for n, act, wt, *_ in ranked[:15]:
        print(f"{n:<55} {act['absmax/std']:>14.1f} {act['ch_ratio']:>12.1f} {act['kurtosis']:>9.1f} "
              f"{wt['absmax/std']:>12.1f} {wt['kurtosis']:>7.1f}")

    _plot(per_layer, ranked[:plot_top], out_dir)
    print(f"\nWrote {out_dir}/linear_stats.csv and plots")
    return rows


def _plot(per_layer, top, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    names = [n for n, *_ in per_layer]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.12), 4))
    ax.plot(x, [a["absmax/std"] for _, a, *_ in per_layer], ".-", label="activation")
    ax.plot(x, [w["absmax/std"] for _, _, w, *_ in per_layer], ".-", label="weight")
    ax.set_yscale("log")
    ax.set_xlabel("Linear layer index (forward order of named_modules)")
    ax.set_ylabel("absmax / std")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120)
    plt.close(fig)

    for n, act, wt, ch_absmax, act_sample, w in top:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        axes[0].bar(range(ch_absmax.numel()), ch_absmax.numpy(), width=1.0)
        axes[0].set_title(f"act per-channel absmax (max/median={act['ch_ratio']:.1f})")
        axes[0].set_xlabel("input channel")
        axes[1].hist(act_sample.numpy(), bins=200, log=True)
        axes[1].set_title(f"act histogram (kurtosis={act['kurtosis']:.1f})")
        axes[2].hist(w.flatten().numpy(), bins=200, log=True)
        axes[2].set_title(f"weight histogram (kurtosis={wt['kurtosis']:.1f})")
        fig.suptitle(n)
        fig.tight_layout()
        idx = names.index(n)
        fig.savefig(os.path.join(out_dir, f"layer_{idx:03d}_{n.replace('.', '_')}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    from models import init_model

    model_ptq = init_model("conformer")
    model_ptq.load()
    model_ptq.model.cuda().eval()
    collect_linear_stats(model_ptq.model, lambda: model_ptq.calib_forward(10),
                         out_dir="assets/dist_stats_conformer")
