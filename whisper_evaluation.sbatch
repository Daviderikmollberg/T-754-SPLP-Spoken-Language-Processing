#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH  --output ./"%x-%j.log"

python3 03-eval_whisper.py --streaming yes --device 0 --language "icelandic" --batch_size 16 --config is_is --dataset <INSERT> --model_id <INSERT>