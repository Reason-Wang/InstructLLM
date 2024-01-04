#!/bin/bash

#SBATCH --job-name=train_llama2_20000_20000_0
#SBATCH --output=/home/xudong.han/llama2/logs/%A_%x.txt
#SBATCH --error=/home/xudong.han/llama2/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=eval
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=40000
#SBATCH --gres=gpu:8
#SBATCH --partition=A100
#SBATCH --qos gpu-8
#SBATCH --time=12:00:00

cd /home/xudong.han/InstructLLM

cmd="deepspeed --num_gpus=8 --launcher SLURM train.py \
  --model_name_or_path meta-llama/Llama-2-13b \
  --deepspeed src/deepspeed_z3_config.json \
  --cache_dir /home/xudong.han/llama2/hf/13B \
  --architecture causal \
  --output_dir /home/xudong.han/llama2/ckpts/llama2_13b_20000_20000_0 \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 20000 \
  --num_code_data 20000 \
  --num_instruction_data 0 \
  --simple_responses False \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --bf16 \
  --logging_steps 10"

echo ${cmd}
eval ${cmd}