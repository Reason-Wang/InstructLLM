#!/bin/bash
#SBATCH --job-name=train_llama2
#SBATCH --output=/home/minghao.wu/llama2/logs/%A-%a.txt
#SBATCH --error=/home/minghao.wu/llama2/logs/%A-%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20000
#SBATCH --qos cpu-4
#SBATCH --time=1:00:00

cd /home/minghao.wu/InstructLLM

cmd="python train.py \
  --model_name_or_path meta-llama/Llama-2-7b \
  --deepspeed src/deepspeed_z3_config.json \
  --cache_dir /l/users/minghao.wu/llama2/hf \
  --architecture causal \
  --output_dir /l/users/minghao.wu/llama2_0_20000_0 \
  --save_strategy no \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing False \
  --num_code_data 20000 \
  --num_instruction_data 0 \
  --simple_responses False \
  --bf16 \
  --logging_steps 50"
echo ${cmd}
eval ${cmd}