import os 
import json
import torch
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pl.callbacks import ModelCheckpoint, RichProgressBar
from pl.loggers import WandbLogger

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

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    #def validation_step(self, batch, batch_idx):
    #    outputs = self.model(**batch)
    #    loss = outputs.loss
    #    self.log("val_loss", loss, prog_bar=True)
    #    return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer

@hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg:DictConfig):
    pl.seed_everything(cfg.seed)

    with open(cfg.data.path, "r") as f:
        data = json.load(f)
    
    def tokenize_function(examples):
        formatted_prompt = [format_prompt(example) for example in examples]
        return tokenizer(
            formatted_prompt, 
            padding="max_length",
            truncation=True,
            max_length=cfg.model.max_seq_length
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)            #contains token ids & attention masks [1s for real tokens, 0s for padding]
    tokenized_dataset.set_format(                                               #converts only input_ids and attention_mask into PyTorch tensors
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=cfg.trainer.per_device_train_batch_size,
        shuffle=True
    )

    model=AstraLightningModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        dir_path=cfg.trainer.output_dir,
        filename='astra-{epoch:02d}-{train_loss:.2f}',
        save_top_k=-1,
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        save_dir=cfg.trainer.output_dir,
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs = cfg.trainer.num_train_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, RichProgressBar()],
        val_check_interval=0.0,
        accumulate_grad_batches=cfg.trainer.gradient_accumulation_steps,
        log_every_n_steps=cfg.trainer.logging_steps,
    )

    print(f"Training {cfg.trainer.num_train_epochs} epochs with {cfg.trainer.batch_size} samples per batch")
    print(f"Using {cfg.trainer.accelerator} with {cfg.trainer.devices} devices")
    #print(f"Using {cfg.trainer.precision} precision")
    #print(f"Using {cfg.trainer.gradient_accumulation_steps} gradient accumulation steps")
    #print(f"Using {cfg.trainer.logging_steps} logging steps")

    trainer.fit(model, train_loader)

    final_adapter_path = os.path.join(cfg.trainer.output_dir, "final_adapter.pt")
    model.model.save_pretrained(final_adapter_path)
    print(f"Final adapter saved to {final_adapter_path}")

    wandb_logger.finish()

if __name__ == "__main__":
    train()