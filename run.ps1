param(
  [string]$Model = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  [int]$Epochs = 2
)
$ErrorActionPreference = "Stop"
$py = ".\venv\Scripts\python.exe"
& $py train.py --model $Model --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw "training failed" }
& $py llama.cpp\convert_hf_to_gguf.py merged --outfile vena-coder-f16.gguf --outtype f16
& $py llama.cpp\convert_hf_to_gguf.py merged --outfile vena-coder-q8_0.gguf --outtype q8_0
if (Get-Command llama-quantize -ErrorAction SilentlyContinue) {
  llama-quantize vena-coder-f16.gguf vena-coder-q4_k_m.gguf Q4_K_M
} else {
  Write-Host "Q8 done. For Q4: winget install ggml.llamacpp, reopen PowerShell, then: llama-quantize vena-coder-f16.gguf vena-coder-q4_k_m.gguf Q4_K_M"
}
