# Required imports
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
import pandas as pd
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Prepare training data
training_data = [
    {"term": "Customer Name", "definition": "The full name of a customer recorded in the system."}
]

# 2. Format data for model training
def format_training_example(term, definition):
    return f"Business Term: {term}\nDefinition: {definition} <|eos|>"

# Create formatted training examples
formatted_data = [format_training_example(item["term"], item["definition"]) 
                 for item in training_data]

# Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_data}))

# 3. Load TinyLLaMA model and tokenizer
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"  # Replace with actual TinyLLaMA if available
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Move model to GPU

# 4. Tokenize the dataset and prepare labels
def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # Create labels by copying input_ids (shifted automatically by the model during training)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# 5. Set up training arguments with CUDA support
training_args = TrainingArguments(
    output_dir="/content/drive/My Drive/model",
   num_train_epochs=5,  # Increased epochs for better learning
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,  # Slightly higher learning rate
    weight_decay=0.01,
    warmup_steps=10,  # Added warmup for stability
    report_to="none",
    fp16=True if torch.cuda.is_available() else False,
)

# 6. Create Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting model training on CUDA...")
trainer.train()
print("Training completed!")

# 7. Save the trained model and tokenizer
output_dir = "/content/drive/My Drive/model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")


def predict_definition(term, max_length=50):
    # Load the trained model and tokenizer
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)  # Move to GPU
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # Prepare input
    input_text = f"Business Term: {term}\nDefinition:"
    inputs = loaded_tokenizer(input_text, return_tensors="pt").to(device)  # Move inputs to GPU
    

    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=loaded_tokenizer.eos_token_id
        )
    
    # Decode and clean the output
    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('generated_text: ',generated_text)
    definition = generated_text.split("Definition:")[1].strip()
    if "<|eos|>" in definition:
        definition = definition.split("<|eos|>")[0].strip()
    return definition

new_terms = [
    "User Name",
]

print("\nusing CUDA:")
for term in new_terms:
    definition = predict_definition(term)
    print(f"Term: {term}")
    print(f"Predicted Definition: {definition}\n")
