import os 
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.finetune import AstraLightningModule

def merge_model(base_model_id: str, checkpoint_path: str, output_path: str):

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        sys.exit(1)

    print(f"Loading base model from {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer from {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )

    print(f"Loading PEFT adapter from checkpoint: {checkpoint_path}...")

    lora_model = AstraLightningModule.load_from_checkpoint(checkpoint_path, strict=False).model
    
    print(f"Merging PEFT adapter with base model...")
    model = lora_model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model successfully merged and saved to {output_path}")

if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print("Usage: python src/merge.py <base_model_id> <path_to_checkpoint.ckpt> <output_path>")
        print("Example: python src/merge.py bigcode/starcoder2-7b outputs/.../best.ckpt models/merged_model")
        sys.exit(1)

    base_model_id_arg = sys.argv[1]
    checkpoint_path_arg = sys.argv[2]
    output_path_arg = sys.argv[3]

    merge_model(base_model_id_arg, checkpoint_path_arg, output_path_arg)