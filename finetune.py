import os 
import json
import torch
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pl.callbacks import ModelCheckpoint

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, DoraConfig
from datasets import Dataset

#This function formats sample required for training
def format_prompt(sample):
    return f""" Instruction: 
    {sample['instruction']}                

    ###Response:
    {sample['output']}"""

class AstraLightningModule(pl.LightningModule):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)      #saves the hyperparameters for easy access
        self.cfg = cfg                      #use self.cfg for easy access

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.id
            trust_remote_code=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.id,
            trust_remote_code=True,
        )

        dora_config = DoraConfig(**self.cfg.peft)
        self.model = get_peft_model(model, dora_config)
        self.model.print_trainable_parameters()

    #The forward pass is used only for inference
    def forward(self, batch):
        return self.model(**batch)