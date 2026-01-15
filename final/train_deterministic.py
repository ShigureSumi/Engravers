import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ================= Configuration =================
# 1. Point to the ORIGINAL Base Model (Unsloth needs this first)
BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

# 2. Path to your EXISTING Fine-tuned Adapter (The Checkpoint)
# This assumes the path on the server.
ADAPTER_PATH = "/root/Engravers/llama3_gold_quant_checkpoint" 

# 3. Data and Output
DATA_FILE = "final/gold_pure_deterministic_train.jsonl"
OUTPUT_DIR = "final/llama3_gold_deterministic_checkpoint"

MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = True

def train():
    print(f"🔥 Starting Deterministic Fine-tuning...")
    print(f"   Base Model: {BASE_MODEL_NAME}")
    print(f"   Adapter:    {ADAPTER_PATH}")
    print(f"   Data:       {DATA_FILE}")
    print(f"   Output:     {OUTPUT_DIR}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load the ORIGINAL Base Model
    print("\n[1/5] Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME, # Load Llama-3 base first
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Load the Existing Adapter (Checkpoint)
    print(f"\n[2/5] Loading Existing Adapter from {ADAPTER_PATH}...")
    try:
        model.load_adapter(ADAPTER_PATH)
        print("   ✅ Adapter loaded successfully!")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not load adapter directly: {e}")
        print("   Continuing with base model only (fresh start)...")

    # 3. Configure LoRA for FURTHER Training
    # We need to ensure the model is in training mode. 
    print("\n[2.5/5] ensuring model is trainable...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 4. Load & Format Data
    print("\n[3/5] Loading & Formatting Data...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Training data file not found: {DATA_FILE}")
        
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token 
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)
    print(f"   Dataset size: {len(dataset)} samples")

    # 5. Initialize Trainer
    print("\n[4/5] Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 16, 
            gradient_accumulation_steps = 1,
            warmup_steps = 10,
            max_steps = 250, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR, 
        ),
    )

    # 6. Train
    print("\n[5/5] Training Started 🚀")
    trainer.train()

    print(f"\n✅ Training Complete!")
    
    # Save Model
    print(f"   Saving new checkpoint to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()