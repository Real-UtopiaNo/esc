#!/usr/bin/env bash
set -euo pipefail
cd /workspace/esc
export PYTHONPATH=/workspace/esc
mkdir -p logs/hif8
python inference.py --model conformer --quant_format hif8 --coverage all --calib max \
  > logs/hif8/conformer_hif8_all.out 2>&1
python inference.py --model conformer --quant_format hif8 --coverage gemm --calib max \
  > logs/hif8/conformer_hif8_gemm.out 2>&1
grep -E 'Average WER|Average CER' logs/hif8/conformer_hif8_all.out logs/hif8/conformer_hif8_gemm.out
