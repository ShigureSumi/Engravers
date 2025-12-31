# ==========================================
# 1. 导入
# ==========================================
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 2. 配置 (RTX 5090 / Ubuntu)
# ==========================================
MAX_SEQ_LENGTH = 4096 
DTYPE = None 
LOAD_IN_4BIT = True 

DATA_FILE = "llama3_finetune_data.jsonl"
OUTPUT_DIR = "llama3_financial_analyst_checkpoint"

print(f"🔥 唤醒 RTX 5090... CUDA: {torch.cuda.is_available()}")

# 3. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", 
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# 4. 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 5. 准备数据
try:
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
except:
    print("⚠️ 没找到数据文件，生成模拟数据测试...")
    from datasets import Dataset
    dataset = Dataset.from_list([{"instruction":"test","input":"test","output":"test"}])

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. 训练器 (Trainer)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 4,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 8, # 安全起见先设8，稳定后再改16
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
    ),
)

# 7. 训练
print("🚀 [Start] 训练开始...")
trainer_stats = trainer.train()

print(f"✅ 训练完成！")
model.save_pretrained(OUTPUT_DIR)