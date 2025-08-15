import os 
import json
import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

@hydra.main(config_path="configs", config_name="config.yaml")
