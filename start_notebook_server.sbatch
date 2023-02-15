#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --gpus-per-task 5
#SBATCH  --output ./"%x-%j.log"
#SBATCH --time 0-07:00:00

. ~/anaconda3/etc/profile.d/conda.sh
conda activate whisper
a_free_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
~/anaconda3/envs/whisper/bin/jupyter notebook --no-browser --port $a_free_port