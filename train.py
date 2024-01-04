import os
from dataclasses import field, dataclass
from typing import Optional, Any
import huggingface_hub

from data.utils import load_bias_data

huggingface_hub.login("")
import transformers
from transformers import Trainer, GPTNeoXTokenizerFast
from data.dataset import Seq2SeqDataset, Seq2SeqCollator, CausalLMDataset, CausalLMCollator
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="google/flan-t5-base")
    architecture: str = field(default='causal')
    data_path: str = field(default="./alpaca_instructions_df.pkl")
    instruction_length: int = 128
    output_length: int = 384
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size = 8
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    num_p3_data: int = 0
    num_code_data: int = 4000
    num_instruction_data: int = None
    simple_responses: bool = False


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # LlamaTokenizer seems not compatible with AutoTokenizer
    if "llama" in args.model_name_or_path.lower():
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=512,
        )
        model = transformers.LlamaForCausalLM.from_pretrained(
            args.cache_dir,
            cache_dir=args.cache_dir,
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=512,
            use_fast=True
        )
        if args.architecture == 'causal':
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
            )
        elif args.architecture == 'seq2seq':
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
            )
        else:
            raise RuntimeError("Architecture must be causal or seq2seq!")

    instructions, responses = load_bias_data()
    # instructions, responses = load_data(args.num_code_data, args.num_instruction_data, args.num_p3_data, args.simple_responses)

    if args.architecture == 'causal':
        dataset = CausalLMDataset(tokenizer, instructions, responses, max_length=512)
        collator = CausalLMCollator(tokenizer, max_length=512)
    elif args.architecture == 'seq2seq':
        dataset = Seq2SeqDataset(instructions, responses)
        collator = Seq2SeqCollator(tokenizer, args.instruction_length, args.output_length)

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)

    # import huggingface_hub
    # from huggingface_hub import create_repo
    # huggingface_hub.login("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")
    # create_repo("reasonwang/"+args.model_name_or_path.replace('/', '-')+"-alpaca")
    # tokenizer.push_to_hub("reasonwang/"+args.model_name_or_path.replace('/', '-')+"-alpaca")
    # model.push_to_hub("reasonwang/" + args.model_name_or_path.replace('/', '-') + "-alpaca")

'''
deepspeed --num_gpus=4 train.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --deepspeed src/deepspeed_z3_config.json \
  --cache_dir /root/autodl-tmp/llama/hf \
  --architecture causal \
  --output_dir /root/autodl-tmp/InstructLLM/ckpts \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 2000 \
  --num_code_data 0 \
  --num_instruction_data 0 \
  --simple_responses False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --bf16 \
  --logging_steps 10
  
python train.py \
  --model_name_or_path gpt2-large \
  --architecture causal \
  --output_dir ckpts/gpt2_medium_simple_4500_4500/ \
  --save_strategy "no" \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --gradient_checkpointing False \
  --num_code_data 4500 \
  --num_instruction_data 4500 \
  --simple_responses True \
  --logging_steps 50
  
python train.py \
  --model_name_or_path distilgpt2 \
  --architecture causal \
  --output_dir ckpts/distilgpt2_0_8000/ \
  --save_strategy "no" \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --num_code_data 0 \
  --num_instruction_data 8000 \
  --logging_steps 50
  
python train.py \
  --model_name_or_path EleutherAI/pythia-70m-deduped \
  --architecture causal \
  --output_dir ckpts/pythia_70m_0_52000/ \
  --save_strategy "no" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --num_code_data 0 \
  --num_instruction_data 52000 \
  --num_train_epochs 2 \
  --logging_steps 50
  
  
python train.py \
  --model_name_or_path EleutherAI/pythia-70m-deduped \
  --architecture causal \
  --output_dir ckpts/pythia_70m_simple_0_8000/ \
  --save_strategy "no" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --num_code_data 0 \
  --num_instruction_data 8000 \
  --simple_responses True \
  --logging_steps 50
  

python train.py \
  --model_name_or_path roneneldan/TinyStories-33M \
  --architecture causal \
  --output_dir ckpts/tinystories_33m_simple_0_8000/ \
  --save_strategy "no" \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --num_code_data 0 \
  --num_instruction_data 8000 \
  --simple_responses True \
  --logging_steps 50
'''

if __name__ == "__main__":
    train()
