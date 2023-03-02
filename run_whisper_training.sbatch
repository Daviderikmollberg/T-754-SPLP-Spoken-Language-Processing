#!/usr/bin/env bash
#SBATCH --mem 10G
#SBATCH --cpus-per-task 5
#SBATCH --output ./"%x-%j.log"
#SBATCH --time 0-07:00:00
#SBATCH --gres=gpu:0


python "./04-FinetuningWhisper.py" \
--model_name_or_path="openai/whisper-tiny" \
--dataset_name="language-and-voice-lab/samromur_asr" \
--language="icelandic" \
--train_split_name="train" \
--model_index_name="Whisper tiy Icelandic" \
--max_steps="1000" \
--output_dir="./whsiper-tiny" \
--per_device_train_batch_size="16" \
--gradient_accumulation_steps="4" \
--logging_steps="25" \
--learning_rate="1e-5" \
--warmup_steps="500" \
--evaluation_strategy="steps" \
--save_strategy="steps" \
--save_steps="250" \
--eval_dataset_name="google/fleurs" \
--eval_dataset_config_name="is_is" \
--eval_split_name="validation" \
--eval_steps="250" \
--do_normalize_eval \
--per_device_eval_batch_size="16" \
--max_eval_samples="100" \
--generation_max_length="225" \
--length_column_name="input_length" \
--max_duration_in_seconds="30" \
--text_column_name="normalized_text" \
--freeze_feature_encoder="False" \
--report_to="tensorboard" \
--metric_for_best_model="wer" \
--greater_is_better="False" \
--load_best_model_at_end \
--gradient_checkpointing \
--fp16 \
--overwrite_output_dir \
--do_train \
--do_eval \
--predict_with_generate \
--streaming \
--push_to_hub=False \

