#!/bin/bash

#SBATCH --job-name=train_llama2_10000_10000_20000
#SBATCH --output=/home/minghao.wu/llama2/logs/%A_%x.txt
#SBATCH --error=/home/minghao.wu/llama2/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=44000
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --qos gpu-8
#SBATCH --time=12:00:00

cd /home/minghao.wu/InstructLLM

cmd="deepspeed --num_gpus=4 --launcher SLURM train.py \
  --model_name_or_path meta-llama/Llama-2-7b \
  --deepspeed src/deepspeed_z3_config.json \
  --cache_dir /l/users/minghao.wu/llama2/hf \
  --architecture causal \
  --output_dir /l/users/minghao.wu/llama2/ckpts/llama2_10000_10000_20000 \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 10000 \
  --num_code_data 10000 \
  --num_instruction_data 20000 \
  --simple_responses False \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --bf16 \
  --logging_steps 10"

echo ${cmd}
eval ${cmd}
