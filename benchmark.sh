#!/bin/bash

string_list=("conformer" "ast" "ecapa" "fastspeech2")
path_list=("scale/conformer_b4_esc.pkl" "scale/ast_b4_esc.pkl" "scale/ecapa_b4_esc.pkl" "scale/fastspeech2_b4_esc.pkl")

for i in "${!string_list[@]}"; do
    model="${string_list[$i]}"
    path="${path_list[$i]}"

    python inference.py --model "$model" --precompute_scale "$path" --ptq bc --num_bits 4
done
