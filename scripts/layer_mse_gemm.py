"""Per-layer output MSE: FP32 vs NVFP4-GEMM and FP32 vs INT4-block16-GEMM (method A).

For each of the 2620 LibriSpeech test-clean utterances, run FP32 and each quantized
model end-to-end, then compare corresponding GEMM-layer outputs. Mean MSE is the
average over utterances of the per-utterance element-wise MSE.
"""
from __future__ import annotations

import csv
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import init_model
from models.conformer.utils import load_featurizer
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from utils.quant_utils import calibrate_model, set_block_quant

OUT_DIR = "assets"
CSV_PATH = f"{OUT_DIR}/layer_mse_gemm.csv"
TABLE_PATH = f"{OUT_DIR}/layer_mse_gemm_table.md"
HEATMAP_PATH = f"{OUT_DIR}/layer_mse_gemm_heatmap.png"
SCATTER_PATH = f"{OUT_DIR}/layer_mse_gemm_scatter.png"
BAR_PATH = f"{OUT_DIR}/layer_mse_gemm_by_type.png"

LAYER_TYPE_ORDER = [
    "ff1.module.1", "ff1.module.4",
    "attention.W_q", "attention.W_k", "attention.W_v", "attention.W_pos", "attention.W_out",
    "conv_block.module.0", "conv_block.module.6",
    "ff2.module.1", "ff2.module.4",
]
LAYER_TYPE_LABELS = {
    "ff1.module.1": "ff1.1", "ff1.module.4": "ff1.4",
    "attention.W_q": "W_q", "attention.W_k": "W_k", "attention.W_v": "W_v",
    "attention.W_pos": "W_pos", "attention.W_out": "W_out",
    "conv_block.module.0": "pw0", "conv_block.module.6": "pw6",
    "ff2.module.1": "ff2.1", "ff2.module.4": "ff2.4",
}


def gemm_layer_names(model):
    """Names of GEMM layers that remain quantized under coverage='gemm'.
    Prefer Quant* modules; fall back to plain nn modules for an FP32 model built
    before quant_modules.initialize() monkey-patches nn.Linear/Conv.
    """
    names = []
    for n, m in model.named_modules():
        if hasattr(m, "input_quantizer"):
            if not m.input_quantizer._disabled and (
                isinstance(m, (quant_nn.QuantLinear, quant_nn.QuantConv1d, quant_nn.QuantConv2d))
            ):
                names.append(n)
            continue
        if isinstance(m, nn.modules.linear.Linear):
            names.append(n)
        elif isinstance(m, (nn.modules.conv.Conv1d, nn.modules.conv.Conv2d)):
            if m.groups == 1 and m.in_channels > 1:
                names.append(n)
    return names


def load_activation_scales(model_ptq, path):
    with open(path, "rb") as f:
        dict_amax = pickle.load(f)
    for name, module in model_ptq.model.named_modules():
        if name in dict_amax and hasattr(module, "input_quantizer") and not module.input_quantizer._disabled:
            amax = dict_amax[name]
            module.input_quantizer.amax = amax.item() if torch.is_tensor(amax) else amax


def build_quant_model(element, scale_path):
    model_ptq = init_model("conformer")
    model_ptq.load()
    set_block_quant(model_ptq, coverage="gemm", element=element)
    # Weight amax from Max calibration; overwrite activation amax with the saved run scales.
    calibrate_model(model_ptq, "conformer", num_calib_batch=100, method="max")
    load_activation_scales(model_ptq, scale_path)
    model_ptq.model.eval().cuda()
    return model_ptq


def attach_hooks(model, layer_names, store):
    modules = dict(model.named_modules())
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            store[name] = out.detach()
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def layer_type_key(name):
    if name.startswith("encoder.layers."):
        # encoder.layers.{i}.{rest}
        rest = name.split(".", 3)[3]
        return rest
    return name


def block_index(name):
    if name.startswith("encoder.layers."):
        return int(name.split(".")[2])
    return None


@torch.no_grad()
def collect_mse(dataset, model_fp32, models_q, layer_names):
    """models_q: dict format -> model. Returns per-format per-layer lists of per-utt MSE/RelMSE."""
    featurizer = load_featurizer()
    mse = {fmt: defaultdict(list) for fmt in models_q}
    rel = {fmt: defaultdict(list) for fmt in models_q}

    fp_store, q_stores = {}, {fmt: {} for fmt in models_q}
    fp_hooks = attach_hooks(model_fp32.model, layer_names, fp_store)
    q_hooks = {fmt: attach_hooks(m.model, layer_names, q_stores[fmt]) for fmt, m in models_q.items()}

    try:
        for waveform, *_ in tqdm(dataset, total=len(dataset), desc="layer MSE"):
            x = featurizer(waveform).cuda().permute(0, 2, 1)
            model_fp32.model(x)
            for fmt, m in models_q.items():
                m.model(x)
                for name in layer_names:
                    y_fp = fp_store[name].float()
                    y_q = q_stores[fmt][name].float()
                    diff = y_q - y_fp
                    mse[fmt][name].append(diff.pow(2).mean().item())
                    denom = y_fp.pow(2).mean().item()
                    rel[fmt][name].append((diff.pow(2).mean().item() / denom) if denom > 0 else 0.0)
    finally:
        remove_hooks(fp_hooks)
        for handles in q_hooks.values():
            remove_hooks(handles)

    return mse, rel


def summarize(mse, rel, layer_names):
    rows = []
    for name in layer_names:
        row = {
            "layer": name,
            "block": block_index(name),
            "type": layer_type_key(name),
            "mse_nvfp4": float(np.mean(mse["nvfp4"][name])),
            "mse_int4_block": float(np.mean(mse["int4_block"][name])),
            "relmse_nvfp4": float(np.mean(rel["nvfp4"][name])),
            "relmse_int4_block": float(np.mean(rel["int4_block"][name])),
        }
        rows.append(row)
    return rows


def write_csv(rows, path):
    fields = ["layer", "block", "type", "mse_nvfp4", "mse_int4_block", "relmse_nvfp4", "relmse_int4_block"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_table(rows, path):
    lines = [
        "# Per-layer mean MSE (GEMM coverage, method A)",
        "",
        "Mean over 2620 LibriSpeech test-clean utterances of the per-utterance element-wise MSE "
        "between each quantized model's layer output and the FP32 baseline's corresponding layer output. "
        "Both models run end-to-end on the same utterance (error accumulates across layers).",
        "",
        "| Layer | FP32 vs NVFP4 GEMM | FP32 vs INT4-block16 GEMM |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| `{r['layer']}` | {r['mse_nvfp4']:.6e} | {r['mse_int4_block']:.6e} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_heatmap(rows, path):
    n_blocks = 16
    types = LAYER_TYPE_ORDER
    mats = {}
    for key, col in [("nvfp4", "mse_nvfp4"), ("int4_block", "mse_int4_block")]:
        mat = np.full((n_blocks, len(types)), np.nan)
        for r in rows:
            if r["block"] is None:
                continue
            if r["type"] in types:
                mat[r["block"], types.index(r["type"])] = r[col]
        mats[key] = mat

    vals = np.concatenate([m[~np.isnan(m)] for m in mats.values()])
    vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    labels = [LAYER_TYPE_LABELS[t] for t in types]
    for ax, (title, mat) in zip(axes, [("NVFP4 GEMM", mats["nvfp4"]), ("INT4-block16 GEMM", mats["int4_block"])]):
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Layer type")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n_blocks))
        ax.set_yticklabels([f"block {i}" for i in range(n_blocks)])
    axes[0].set_ylabel("Conformer block")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="mean MSE")
    fig.suptitle("Per-layer mean MSE vs FP32 (GEMM layers in blocks)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(rows, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.array([r["mse_nvfp4"] for r in rows])
    y = np.array([r["mse_int4_block"] for r in rows])
    ax.scatter(x, y, s=18, alpha=0.7, edgecolors="none")
    lim = (min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2)
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean MSE: FP32 vs NVFP4 GEMM")
    ax.set_ylabel("mean MSE: FP32 vs INT4-block16 GEMM")
    ax.set_title("Per-layer mean MSE (179 GEMM layers)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_type(rows, path):
    # Mean over the 16 blocks for each in-block type; singleton layers listed separately.
    by_type = defaultdict(lambda: {"nvfp4": [], "int4_block": []})
    singles = []
    for r in rows:
        if r["block"] is None:
            singles.append(r)
        else:
            by_type[r["type"]]["nvfp4"].append(r["mse_nvfp4"])
            by_type[r["type"]]["int4_block"].append(r["mse_int4_block"])

    labels, nv, ib = [], [], []
    for t in LAYER_TYPE_ORDER:
        labels.append(LAYER_TYPE_LABELS[t])
        nv.append(float(np.mean(by_type[t]["nvfp4"])))
        ib.append(float(np.mean(by_type[t]["int4_block"])))
    for r in singles:
        labels.append(r["layer"].split(".")[-1] if "decoder" in r["layer"] or "linear_proj" in r["layer"]
                      else r["layer"].replace("encoder.", ""))
        nv.append(r["mse_nvfp4"])
        ib.append(r["mse_int4_block"])

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - width / 2, nv, width, label="NVFP4 GEMM")
    ax.bar(x + width / 2, ib, width, label="INT4-block16 GEMM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean MSE vs FP32")
    ax.set_title("Mean MSE by layer type (block types averaged over 16 blocks)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def topk_lines(rows, k=10):
    lines = ["### Top-10 layers by mean MSE", ""]
    for fmt, col, title in [
        ("nvfp4", "mse_nvfp4", "FP32 vs NVFP4 GEMM"),
        ("int4_block", "mse_int4_block", "FP32 vs INT4-block16 GEMM"),
    ]:
        ranked = sorted(rows, key=lambda r: r[col], reverse=True)[:k]
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| Rank | Layer | mean MSE |")
        lines.append("|---:|---|---:|")
        for i, r in enumerate(ranked, 1):
            lines.append(f"| {i} | `{r['layer']}` | {r[col]:.6e} |")
        lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building FP32 model...")
    model_fp32 = init_model("conformer")
    model_fp32.load()
    model_fp32.model.eval().cuda()

    print("Building NVFP4 / INT4-block GEMM models...")
    quant_modules.initialize()
    try:
        model_nv = build_quant_model("e2m1", "scale/conformer_nvfp4_gemm_max.pkl")
        model_ib = build_quant_model("int4", "scale/conformer_int4_block_gemm_max.pkl")
    finally:
        # Keep patches active while models exist; deactivate after collection.
        pass

    layer_names = gemm_layer_names(model_nv.model)
    assert layer_names == gemm_layer_names(model_ib.model), "NVFP4 and INT4-block GEMM layer sets differ"
    fp32_modules = dict(model_fp32.model.named_modules())
    missing = [n for n in layer_names if n not in fp32_modules]
    assert not missing, f"FP32 model missing layers: {missing[:5]}"
    print(f"GEMM layers: {len(layer_names)}")

    dataset = model_fp32.val_dataset
    print(f"Utterances: {len(dataset)}")

    mse, rel = collect_mse(
        dataset, model_fp32,
        {"nvfp4": model_nv, "int4_block": model_ib},
        layer_names,
    )
    quant_modules.deactivate()

    rows = summarize(mse, rel, layer_names)
    write_csv(rows, CSV_PATH)
    write_table(rows, TABLE_PATH)
    plot_heatmap(rows, HEATMAP_PATH)
    plot_scatter(rows, SCATTER_PATH)
    plot_by_type(rows, BAR_PATH)

    with open(f"{OUT_DIR}/layer_mse_gemm_topk.md", "w") as f:
        f.write(topk_lines(rows))

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {HEATMAP_PATH}, {SCATTER_PATH}, {BAR_PATH}")
    print(topk_lines(rows))


if __name__ == "__main__":
    main()
