$ErrorActionPreference = "Stop"
if (-not (Test-Path venv)) { python -m venv venv }
$py = ".\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install transformers peft datasets accelerate sentencepiece gguf numpy protobuf
if (-not (Test-Path llama.cpp)) { git clone --depth 1 https://github.com/ggml-org/llama.cpp }
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available())"
