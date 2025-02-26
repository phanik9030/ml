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


from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
import pandas as pd
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

output_dir = "/content/drive/My Drive/trained_business_terms_model"

# 8. Predict functions with improved cleaning
def predict_definition(term, max_length=256):
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    input_text = f"Business Term: {term}\nDefinition:"
    inputs = loaded_tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.2,  # Lowered for more coherence
            top_p=0.85,       # Tightened for less randomness
            do_sample=True,
            pad_token_id=loaded_tokenizer.eos_token_id,
            eos_token_id=loaded_tokenizer.convert_tokens_to_ids('<|eos|>'),
        )
    
    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        definition = generated_text.split("Definition:")[1].split('<|eos|>')[0].strip()
        # Remove any trailing nonsense
        definition = ' '.join(definition.split()[:10])  # Limit to first 10 words
        return definition
    except IndexError:
        return "Failed to generate a valid definition."

def predict_associated_term(column, max_length=50):
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    input_text = f"Column: {column}\nAssociatedTerm:"
    inputs = loaded_tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.80,
            do_sample=True,
            pad_token_id=loaded_tokenizer.eos_token_id,
            eos_token_id=loaded_tokenizer.convert_tokens_to_ids('<|eos|>'),
        )
    
    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        associated_term = generated_text.split("AssociatedTerm:")[1].split('<|eos|>')[0].strip()
        # Remove any trailing nonsense
        associated_term = ' '.join(associated_term.split()[:5])  # Limit to first 5 words
        return associated_term
    except IndexError:
        return "Failed to generate a valid associated term."

# 9. Test
new_terms = ["Cash Flow Year over Year Change Percentage", "Appraisal Basic Valuation Type", "Vendor Name"]
new_columns = ["xfer_mthd_sub", "rgn_id", "target_bal"]

print("\nPredicting definitions for new terms using CUDA:")
for term in new_terms:
    definition = predict_definition(term)
    print(f"Term: {term}")
    print(f"Predicted Definition: {definition}\n")

print("Predicting associated terms for new columns using CUDA:")
for column in new_columns:
    associated_term = predict_associated_term(column)
    print(f"Column: {column}")
    print(f"Predicted Associated Term: {associated_term}\n")

