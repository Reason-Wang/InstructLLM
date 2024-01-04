#!/bin/bash

#SBATCH --job-name=train_pythia_0_20000_20000
#SBATCH --output=/home/minghao.wu/llama2/logs/%A_%x.txt
#SBATCH --error=/home/minghao.wu/llama2/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos gpu-8
#SBATCH --time=12:00:00

cd /home/minghao.wu/InstructLLM

cmd="python train.py \
  --model_name_or_path EleutherAI/pythia-1.4b-deduped \
  --architecture causal \
  --output_dir /l/users/minghao.wu/pythia/ckpts/pythia1.4b_0_20000_20000 \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 0 \
  --num_code_data 20000 \
  --num_instruction_data 20000 \
  --simple_responses False \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --logging_steps 10"

echo ${cmd}
eval ${cmd}