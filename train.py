import argparse, glob, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--maxlen", type=int, default=768)
ap.add_argument("--data", default="data/*.jsonl")
a = ap.parse_args()

SYS = "You are Vena, a coding assistant. Answer with correct, concise code and explanations."
tok = AutoTokenizer.from_pretrained(a.model)

texts = []
for fp in glob.glob(a.data):
    for line in open(fp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)["text"]
        except Exception:
            continue
        m = re.match(r"\s*Instruction:\s*(.*?)\s*\nResponse:\s*(.*)", t, re.S)
        if not m:
            continue
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": m.group(1)},
                {"role": "assistant", "content": m.group(2).strip()}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False))

print("examples:", len(texts))
assert texts, "no valid examples found in data/*.jsonl"

class DS(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.enc = [tok(t, truncation=True, max_length=a.maxlen) for t in texts]
    def __len__(self):
        return len(self.enc)
    def __getitem__(self, i):
        return {"input_ids": self.enc[i]["input_ids"], "attention_mask": self.enc[i]["attention_mask"]}

assert torch.cuda.is_available(), "CUDA GPU not found - run setup.ps1 first"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="lora_out", per_device_train_batch_size=1, gradient_accumulation_steps=16,
    num_train_epochs=a.epochs, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, logging_steps=10, save_strategy="no", report_to="none",
    remove_unused_columns=False, optim="adamw_torch")

Trainer(model=model, args=args, train_dataset=DS(texts),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

merged = model.merge_and_unload()
merged.save_pretrained("merged")
tok.save_pretrained("merged")
print("saved -> merged")
