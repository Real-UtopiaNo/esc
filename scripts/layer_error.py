"""Layer-wise and end-to-end error of a quantized model against its baseline (FP32) model.

For every hooked layer, on every input:
  local error        q_layer(fp_input) vs fp_layer(fp_input): the error this layer introduces by itself
  accumulated error  q_model's layer output vs fp_model's layer output for the same model input:
                     local error plus everything propagated from upstream layers
End to end: output_fn(q_model(x)) vs output_fn(fp_model(x)), per input.
With metric_fn, each input also gets a task metric for both models, and the script reports how
the per-input end-to-end error correlates with the per-input metric drop.

fp_model and q_model must be two separate instances of the same architecture (same module names):
build and load the checkpoint twice, quantize (and run HiGPTQ on) only q_model.

Usage:
    from layer_error import compare_models
    compare_models(fp_model, q_model, inputs, out_dir="layer_error",
                   metric_fn=lambda i, y_fp, y_q: (score(y_fp, i), score(y_q, i)))

Outputs in out_dir:
    layer_error.csv   per layer: local / accumulated MSE, relative MSE, SQNR (dB)
    per_input.csv     per input: end-to-end MSE / SQNR, metric of both models and its delta
    layer_error.png   local and accumulated SQNR per layer
    e2e_vs_metric.png end-to-end SQNR vs metric delta per input (with metric_fn)
"""
from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn as nn


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


def _sqnr_db(rel):
    return -10.0 * math.log10(rel) if rel > 0 else math.inf


class _Err:
    def __init__(self):
        self.mse, self.rel = [], []

    def add(self, y, ref):
        y, ref = y.detach().float(), ref.detach().float()
        mse = (y - ref).pow(2).mean().item()
        power = ref.pow(2).mean().item()
        self.mse.append(mse)
        self.rel.append(mse / power if power > 0 else 0.0)

    def summary(self):
        if not self.mse:
            return math.nan, math.nan, math.nan
        mse = sum(self.mse) / len(self.mse)
        rel = sum(self.rel) / len(self.rel)
        return mse, rel, _sqnr_db(rel)


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va > 0 and vb > 0 else math.nan


@torch.no_grad()
def compare_models(fp_model, q_model, inputs, out_dir="layer_error", layer_types=(nn.Linear,),
                   output_fn=None, metric_fn=None, metric_name="metric"):
    """inputs: iterable of model inputs; a tuple is unpacked as positional arguments.
    output_fn(out) -> tensor compared end to end (default: first tensor of the output).
    metric_fn(i, fp_out, q_out) -> (fp_metric, q_metric) for input i (optional).
    """
    os.makedirs(out_dir, exist_ok=True)
    output_fn = output_fn or _first_tensor
    fp_model.eval()
    q_model.eval()

    q_modules = dict(q_model.named_modules())
    names = [n for n, m in fp_model.named_modules() if isinstance(m, layer_types) and n in q_modules]
    local = {n: _Err() for n in names}
    accum = {n: _Err() for n in names}
    fp_outs = {}
    state = {"q_pass": False}

    def fp_hook(n):
        def hook(_mod, args, kwargs, out):
            ref = _first_tensor(out)
            fp_outs.setdefault(n, []).append(ref.detach())
            local[n].add(_first_tensor(q_modules[n](*args, **kwargs)), ref)
        return hook

    def q_hook(n):
        def hook(_mod, _args, out):
            refs = fp_outs.get(n)
            if state["q_pass"] and refs:
                accum[n].add(_first_tensor(out), refs.pop(0))
        return hook

    fp_mods = dict(fp_model.named_modules())
    handles = [fp_mods[n].register_forward_hook(fp_hook(n), with_kwargs=True) for n in names]
    handles += [q_modules[n].register_forward_hook(q_hook(n)) for n in names]

    rows = []
    try:
        for i, x in enumerate(inputs):
            args = x if isinstance(x, tuple) else (x,)
            fp_outs.clear()
            y_fp = fp_model(*args)
            state["q_pass"] = True
            y_q = q_model(*args)
            state["q_pass"] = False

            e = _Err()
            e.add(output_fn(y_q), output_fn(y_fp))
            row = {"idx": i, "e2e_mse": e.mse[0], "e2e_rel": e.rel[0], "e2e_sqnr_db": _sqnr_db(e.rel[0])}
            if metric_fn is not None:
                m_fp, m_q = metric_fn(i, y_fp, y_q)
                row.update({f"{metric_name}_fp": m_fp, f"{metric_name}_q": m_q,
                            f"{metric_name}_delta": m_q - m_fp})
            rows.append(row)
    finally:
        state["q_pass"] = False
        for h in handles:
            h.remove()

    if not rows:
        print("WARNING: inputs is empty; nothing written")
        return [], []

    layer_rows = []
    for n in names:
        lm, lr, ls = local[n].summary()
        am, ar, as_ = accum[n].summary()
        layer_rows.append({"layer": n, "calls": len(local[n].mse),
                           "local_mse": lm, "local_rel": lr, "local_sqnr_db": ls,
                           "accum_mse": am, "accum_rel": ar, "accum_sqnr_db": as_})

    _write_csv(os.path.join(out_dir, "layer_error.csv"), layer_rows)
    _write_csv(os.path.join(out_dir, "per_input.csv"), rows)
    _report(layer_rows, rows, metric_fn is not None, metric_name)
    _plot(layer_rows, rows, metric_fn is not None, metric_name, out_dir)
    print(f"Wrote {os.path.abspath(out_dir)}")
    return layer_rows, rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _report(layer_rows, rows, has_metric, metric_name):
    print(f"\nTop layers by local error (error introduced by the layer itself):")
    print(f"{'layer':<55} {'local SQNR dB':>13} {'accum SQNR dB':>13}")
    for r in sorted(layer_rows, key=lambda r: r["local_sqnr_db"])[:15]:
        print(f"{r['layer']:<55} {r['local_sqnr_db']:>13.2f} {r['accum_sqnr_db']:>13.2f}")

    sqnr = [r["e2e_sqnr_db"] for r in rows]
    finite = [s for s in sqnr if math.isfinite(s)]
    print(f"\nEnd to end over {len(rows)} inputs: mean SQNR {sum(finite) / max(len(finite), 1):.2f} dB, "
          f"mean relative MSE {sum(r['e2e_rel'] for r in rows) / len(rows):.4g}")
    if has_metric and len(rows) > 2:
        fp = [r[f"{metric_name}_fp"] for r in rows]
        q = [r[f"{metric_name}_q"] for r in rows]
        d = [r[f"{metric_name}_delta"] for r in rows]
        pairs = [(s, dd) for s, dd in zip(sqnr, d) if math.isfinite(s)]
        s_, d_ = [p[0] for p in pairs], [p[1] for p in pairs]
        print(f"{metric_name}: baseline {sum(fp) / len(fp):.4f}, quantized {sum(q) / len(q):.4f}, "
              f"mean delta {sum(d) / len(d):+.4f}")
        print(f"corr(e2e SQNR, {metric_name} delta): Pearson {_pearson(s_, d_):+.3f}, "
              f"Spearman {_pearson(_rank(s_), _rank(d_)):+.3f}")


def _plot(layer_rows, rows, has_metric, metric_name, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    def clean(v):
        return v if math.isfinite(v) else float("nan")

    x = range(len(layer_rows))
    fig, ax = plt.subplots(figsize=(max(8, len(layer_rows) * 0.12), 4))
    ax.plot(x, [clean(r["local_sqnr_db"]) for r in layer_rows], ".-", label="local (layer alone)")
    ax.plot(x, [clean(r["accum_sqnr_db"]) for r in layer_rows], ".-", label="accumulated")
    ax.set_xlabel("layer index (named_modules order)")
    ax.set_ylabel("SQNR vs baseline (dB, higher is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_error.png"), dpi=120)
    plt.close(fig)

    if has_metric:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([clean(r["e2e_sqnr_db"]) for r in rows], [r[f"{metric_name}_delta"] for r in rows], s=10)
        ax.set_xlabel("end-to-end SQNR (dB)")
        ax.set_ylabel(f"{metric_name} quantized - baseline")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "e2e_vs_metric.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jiwer import wer

    from models import init_model
    from models.conformer.utils import load_featurizer
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from utils.quant_utils import set_block_quant

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    quant_modules.initialize()
    fp = init_model("conformer")
    fp.load()
    for m in fp.model.modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            m.disable()
    q = init_model("conformer")
    q.load()
    set_block_quant(q, coverage="gemm", element="hif4")
    fp.model.cuda()
    q.model.cuda()

    featurizer = load_featurizer()
    inputs, refs = [], []
    for i, (waveform, _sr, reference, *_) in enumerate(fp.val_dataset):
        if i >= num:
            break
        inputs.append(featurizer(waveform).cuda().permute(0, 2, 1))
        refs.append(reference.lower())

    def metric_fn(i, y_fp, y_q):
        def utt_wer(emission):
            return wer(refs[i], " ".join(fp.decoder(emission[0])).lower())
        return utt_wer(y_fp), utt_wer(y_q)

    compare_models(fp.model, q.model, inputs, out_dir="assets/layer_error_conformer_hif4_gemm",
                   metric_fn=metric_fn, metric_name="wer")
