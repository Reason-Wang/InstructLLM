#!/bin/bash
#SBATCH --output=/home/minghao.wu/llama2/logs/%A-%a.txt
#SBATCH --error=/home/minghao.wu/llama2/logs/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos cpu-4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=20000
#SBATCH --job-name=convert_llama_weights

cd /home/minghao.wu/InstructLLM

python -m src.convert_llama_weights_to_hf --input_dir /l/users/minghao.wu/llama2/meta --model_size 7B --output_dir /l/users/minghao.wu/llama2/hf