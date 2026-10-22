# vena-coder
LoRA fine-tune of a small pretrained coder, exported to GGUF for phones.

Data: put JSONL files in `data/` ({"text":"Instruction: ...\nResponse: ..."}).

Windows (NVIDIA GPU):
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    powershell -ExecutionPolicy Bypass -File .\run.ps1
    # smaller GPU: .\run.ps1 -Model Qwen/Qwen2.5-Coder-0.5B-Instruct

Output: vena-coder-q8_0.gguf (and vena-coder-q4_k_m.gguf if llama-quantize is installed).
