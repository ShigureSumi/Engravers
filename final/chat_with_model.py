import torch
from unsloth import FastLanguageModel
import os
import argparse

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

DEFAULT_MODEL_PATH = "/home/dragon/AI/llama-3-8B-4bit-finance"

def chat_loop():
    print(f"Loading Model from {DEFAULT_MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL_PATH,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    print("\n✅ Model Loaded! You can now chat with the FinLLM.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("\n[User]: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Use the Alpaca format which the model was likely trained on or expects
            prompt = f"""### Instruction:
You are a financial AI assistant. Answer the following question.

### Input:
{user_input}

### Response:
"""
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True,
                temperature=0.7, # Add some creativity for chat
                do_sample=True
            )
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the response part
            response_text = response.split("### Response:")[-1].strip()
            
            print(f"\n[FinLLM]: {response_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
