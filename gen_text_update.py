# Required imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
import os
from typing import List, Dict
from torch.optim import AdamW

# Verify CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# 1. Define a custom dataset class for batch loading
class BusinessTermsDataset(TorchDataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 12):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if "term" in item and "definition" in item:
            text = f"Business Term: {item['term']}\nDefinition: {item['definition']} <|eos|>"
        elif "column" in item and "associatedTerm" in item:
            text = f"Column: {item['column']}\nAssociatedTerm: {item['associatedTerm']} <|eos|>"
        else:
            raise ValueError(f"Invalid item at index {idx}: {item}")
        
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze()
        }

# 2. Load your large training data (replace with your actual 20,000+ items)
# Example: loading from a CSV or list
# training_data = pd.read_csv("your_data_file.csv").to_dict("records")
# For demo, using a small subset repeated
training_data = [
    {"term": "Customer Name", "definition": "The full name of a customer recorded in the system."},
    {"term": "Order Date", "definition": "The date an order was placed in the system."},
    {"column": "customer_id", "associatedTerm": "Customer Identifier"},
] * 10000  # Simulating 20,000+ items; replace with your data

dataset_size = len(training_data)
print(f"Total training items: {dataset_size}")

# 3. Load model and tokenizer
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['<|eos|>']})

model = AutoModelForCausalLM.from_pretrained(model_name).to(device).half()  # FP16 from the start
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
if torch.cuda.is_available():
    print(f"After model load - GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"After model load - GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# 4. Create dataset and DataLoader
dataset = BusinessTermsDataset(training_data, tokenizer, max_length=12)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Training parameters
num_epochs = 1
learning_rate = 5e-5
gradient_accumulation_steps = 16
total_steps = (dataset_size // (batch_size * gradient_accumulation_steps)) + 1
optimizer = AdamW(model.parameters(), lr=learning_rate)
warmup_steps = 100
step = 0

# 6. Manual training loop
print("Starting model training on CUDA...")
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    accumulated_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps  # Normalize loss for accumulation
        loss.backward()

        accumulated_loss += loss.item()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # Learning rate warmup
            if step <= warmup_steps:
                lr_scale = float(step) / float(max(1, warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * lr_scale

            if step % 200 == 0:
                print(f"Step {step}/{total_steps}, Loss: {accumulated_loss:.4f}")
                accumulated_loss = 0
                if torch.cuda.is_available():
                    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

            if step % 1000 == 0:
                output_dir = f"./trained_business_terms_model_step_{step}"
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Checkpoint saved at step {step} to {output_dir}")

# 7. Save final model
output_dir = "./trained_business_terms_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training completed! Model saved to {output_dir}")

# Clear memory
del model, optimizer, dataloader, dataset
torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Final GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
