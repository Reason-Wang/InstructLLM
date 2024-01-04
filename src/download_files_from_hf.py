import huggingface_hub
from huggingface_hub import hf_hub_download

huggingface_hub.login("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")

hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="params.json",
    local_dir="./",
)