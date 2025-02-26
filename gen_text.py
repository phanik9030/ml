# Required imports
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
import pandas as pd
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# 1. Prepare training data with additional examples for better generalization
training_data = [
    {"term": "Customer Name", "definition": "The full name of a customer recorded in the system."},
    {"column": "region_code", "associatedTerm": "Regional Code"}
]

# 2. Format data based on type
def format_training_example(term, definition):
    return f"Business Term: {term}\nDefinition: {definition} <|eos|>"

def format_training_example_column(column, associatedTerm):
    return f"Column: {column}\nAssociatedTerm: {associatedTerm} <|eos|>"

# Process data into formatted examples
formatted_data = []
for item in training_data:
    if "term" in item and "definition" in item:
        formatted_data.append(format_training_example(item["term"], item["definition"]))
    elif "column" in item and "associatedTerm" in item:
        formatted_data.append(format_training_example_column(item["column"], item["associatedTerm"]))

# Print formatted data for verification
print("Formatted training data:")
for line in formatted_data:
    print(line)

# Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_data}))

# 3. Load model and tokenizer
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['<|eos|>']})

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))

# 4. Tokenize with error checking
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    assert not torch.any(tokenized["input_ids"] < 0), "Negative token IDs detected"
    assert tokenized["input_ids"].shape == tokenized["labels"].shape, "Shape mismatch"
    return {k: v.squeeze() for k, v in tokenized.items()}

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

print("Sample tokenized data:", tokenized_dataset[0])

# 5. Training arguments with increased epochs
training_args = TrainingArguments(
    output_dir="/content/drive/My Drive/trained_business_terms_model",
    num_train_epochs=20,  # Increased for better learning
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=10,
    report_to="none",
    fp16=True if torch.cuda.is_available() else False,
)

# 6. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting model training on CUDA...")
try:
    trainer.train()
    print("Training completed!")
except RuntimeError as e:
    print(f"Training failed with CUDA error: {e}")
    raise

# 7. Save model
output_dir = "/content/drive/My Drive/trained_business_terms_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

