import os 
import json
import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

@hydra.main(config_path="../configs", config_name="config.yaml")
def merge(cfg:DictConfig):
    base_model_id=cfg.model.id
    
    adapter_path=os.path.join(cfg.trainer.output_dir, "final_adapter")
    merged_model_path=os.path.join(cfg.trainer.output_dir, "merged_model")
    
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

    print(f"Loading PEFT adapter from {adapter_path}...")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path {adapter_path} does not exist")
    
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )
    
    print(f"Merging PEFT adapter with base model...")
    model=model.merge_and_unload()

    print(f"Saving merged model to {merged_model_path}...")
    os.makedirs(
        merged_model_path, 
        exist_ok=True
    )
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    
    print(f"Model saved to {merged_model_path}")

if __name__ == "__main__":
    merge()