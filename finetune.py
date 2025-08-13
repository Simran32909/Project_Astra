import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, DoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

def format_prompt(sample):
    """Formats a sample to be used for training."""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""


def finetune():
    with open('data/core_dataset.json', 'r') as f:
        data=json.load(f)
    
    dataset=Dataset.from_list(data)

    model_id="bigcode/starcoder2-7b"

    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer.pad_token=tokenizer.eos_token
    model.config.use_cache=False

    dora_config=DoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_dora=True,
    )


if __name__ == "__main__":
    finetune()