#SBATCH --gres gpu:1
#SBATCH  --output ./"%x-%j.log"


./03-eval_whisper.py --streaming --device 0 --dataset <INSERT> --model_id <INSERT> --language "is" --batch_size 16 --config is --dataset google/fleurs