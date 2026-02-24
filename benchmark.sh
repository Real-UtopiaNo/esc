#!/bin/bash

string_list=("conformer" "ast" "ecapa" "fastspeech2" "mpnet")
bit_range=(4 8)

#Calibration benchmarking (Table 1)
percentile_list=(99.9 99.99 99.999)
for str in "${string_list[@]}"; do
    for percentile in "${percentile_list[@]}"; do
        for num in "${bit_range[@]}"; do
            python inference.py --model $str --num_bits $num --calib percentile --percentile $percentile
        done
    done 
done

for str in "${string_list[@]}"; do
    for num in "${bit_range[@]}"; do
        python inference_ptq_test.py --model $str --ptq hyq --num_bits $num --calib entropy
        #since esc is based on mse calibration, we can do both at same time
        python inference_ptq_test.py --model $str --ptq hyq --num_bits $num --calib mse --esc
    done
done

#PTQ benchmarking (Table 2)
path_list=("scale/conformer_b4_esc.pkl" "scale/ast_b4_esc.pkl" "scale/ecapa_b4_esc.pkl" "scale/fastspeech2_b4_esc.pkl" "scale/mpnet_b4_esc.pkl")

for i in "${!string_list[@]}"; do
    model="${string_list[$i]}"
    path="${path_list[$i]}"

    python inference.py --model "$model" --precompute_scale "$path" --ptq adaround --num_bits 4 --iter_adaround 500
    python inference.py --model "$model" --precompute_scale "$path" --ptq noisyquant --num_bits 4
    python inference.py --model "$model" --precompute_scale "$path" --ptq ditas --num_bits 4 --iter_rank_ditas 10 --reduced_rank_ditas 32 --grid_ditas 31
    python inference.py --model "$model" --precompute_scale "$path" --ptq hyq --num_bits 4
    python inference.py --model "$model" --precompute_scale "$path" --ptq brecq --num_bits 4 --iter_brecq 200 --lr_brecq 1e-4
    python inference.py --model "$model" --precompute_scale "$path" --ptq smoothquant --num_bits 4
    python inference.py --model "$model" --precompute_scale "$path" --ptq bc --num_bits 4
done
