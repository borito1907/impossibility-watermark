#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cuda_device> <index> <save_name> <attack_num>"
    exit 1
fi

cuda_device=$1
i=$2
save_name="${3}_${i}_${attack_num}"
attack_num=$4

python -m attack attack_args.cuda=\'$cuda_device$\' attack_args.json_index="$i" attack_args.save_name="$save_name" attack_args.is_completion=True attack_args.json_path='./text_completions_50_c4.json' &> completion_"$i"_"$attack_num".txt