from transformers import AutoModelForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
model = AutoModelForCausalLM.from_pretrained("/l/users/minghao.wu/llama2/hf")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)