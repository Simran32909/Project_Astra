import os
import json
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from datasets import Dataset
from bitsandbytes.optim import PagedAdamW8bit

def format_prompt(sample):
    """Formats a sample to be used for training."""
    return f"""### Instruction:
{sample['instruction']}

### Response:
"""

def format_full_prompt(sample):
    """Formats the full prompt including the output for tokenization."""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""


# --- The Core: Pytorch Lightning Module ---
class AstraLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.id, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model with Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.id, 
            quantization_config=bnb_config, 
            #device_map={"": 3},
            trust_remote_code=True
        )
        
        # 3. Apply PEFT (LoRA) and Gradient Checkpointing
        lora_config = LoraConfig(**OmegaConf.to_container(self.cfg.peft))
        self.model = get_peft_model(model, lora_config)
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

    def training_step(self, batch, batch_idx):
        batch["labels"] = batch["input_ids"].clone()
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate validation loss
        batch["labels"] = batch["input_ids"].clone()
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        # --- RECONSTRUCT SAMPLES FROM BATCH ---
        num_samples = len(batch['original_data']['instruction'])
        reconstructed_samples = [
            {'instruction': batch['original_data']['instruction'][i], 'output': batch['original_data']['output'][i]}
            for i in range(num_samples)
        ]
        
        # --- CALCULATE EXACT MATCH (EM) ACCURACY ---
        # 1. Generate predictions from input_ids
        prompts = [format_prompt(s) for s in reconstructed_samples]
        
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=50,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 2. Decode predictions and labels
        preds = []
        for i, ids in enumerate(generated_ids):
            pred_ids = ids[batch['input_ids'][i].size(0):]
            preds.append(self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip())
        
        labels = [s['output'] for s in reconstructed_samples]

        # 3. Compare and calculate accuracy
        exact_matches = sum(1 for pred, label in zip(preds, labels) if pred == label)
        
        # Log individual predictions for debugging
        if batch_idx == 0 and self.global_rank == 0: # Log only on the main process
            for i in range(min(3, len(preds))):
                self.logger.experiment.log({
                    f"val_sample_{i}": wandb.Table(
                        columns=["Instruction", "Prediction", "Label"],
                        data=[[reconstructed_samples[i]['instruction'], preds[i], labels[i]]]
                    )
                })

        output = {'val_loss': val_loss, 'exact_matches': exact_matches, 'total': len(labels)}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_start(self):
        # In modern PyTorch Lightning, we manually create a list to collect outputs
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        # The hook signature has changed. We process the list we collected.
        outputs = self.validation_step_outputs
        if not outputs:
            return

        # Aggregate metrics from all batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        total_matches = sum(x['exact_matches'] for x in outputs)
        total_samples = sum(x['total'] for x in outputs)
        em_accuracy = total_matches / total_samples
        
        # Log epoch-level metrics
        self.log("val_loss_epoch", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_em_accuracy", em_accuracy, on_epoch=True, prog_bar=True)

        # Free memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = PagedAdamW8bit(self.parameters(), lr=self.cfg.trainer.learning_rate)
        return optimizer

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_and_prepare(dataset_path, is_train=True):
        full_path = os.path.join(get_original_cwd(), dataset_path)
        with open(full_path, "r") as f:
            data = json.load(f)
        
        def tokenize_function(examples):
            num_examples = len(examples['instruction'])
            reconstructed_samples = [{'instruction': examples['instruction'][i], 'output': examples['output'][i]} for i in range(num_examples)]
            
            if is_train:
                prompts = [format_full_prompt(s) for s in reconstructed_samples]
            else:
                prompts = [format_prompt(s) for s in reconstructed_samples]

            tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=cfg.model.max_seq_length)
            return tokenized

        dataset = Dataset.from_list(data)
        # The `remove_columns` parameter tells .map to drop the original columns
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=list(data[0].keys()))

        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # We add the original data back in a new column for use in the validation_step
        tokenized_dataset = tokenized_dataset.add_column("original_data", data)

        return tokenized_dataset

    train_dataset = tokenize_and_prepare(cfg.data.train_path, is_train=True)
    val_dataset = tokenize_and_prepare(cfg.data.val_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.per_device_train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.per_device_train_batch_size, num_workers=4)
    
    model = AstraLightningModule(cfg)
    
    save_dir = HydraConfig.get().runtime.output_dir
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = WandbLogger(
        project=cfg.logger.wandb.project,
        name=cfg.logger.wandb.name,
        entity=cfg.logger.wandb.entity,
        log_model=cfg.logger.wandb.log_model,
        save_dir=save_dir,
        config=config_dict
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename='astra-{epoch:02d}-{val_loss:.2f}-{val_em_accuracy:.2f}',
        save_top_k=3,
        monitor="val_em_accuracy",
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.num_train_epochs,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, RichProgressBar()],
        log_every_n_steps=cfg.trainer.logging_steps,
        strategy=cfg.trainer.strategy,
        logger=logger
    )

    try:
        # 4. Train
        print("Starting training and validation...")
        trainer.fit(model, train_loader, val_loader)
        print("Training complete.")

        # 5. Save final adapter
        final_adapter_path = os.path.join(save_dir, "final_adapter")
        model.model.save_pretrained(final_adapter_path)
        print(f"Final adapter saved to {final_adapter_path}")

    finally:
        print("Finishing W&B run...")
        wandb.finish()


if __name__ == "__main__":
    train()