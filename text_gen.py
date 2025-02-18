import json

# Load dataset
with open("business_terms.json", "r") as f:
    data = json.load(f)

# Prepare inputs and labels
inputs = [item["business_term"] for item in data]
labels = [item["definition"] for item in data]

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLaMA 3 8B model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize inputs and labels
input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
label_ids = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids

import torch
from transformers import AdamW

# Fine-tuning loop
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(3):  # 3 epochs
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=label_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model and tokenizer
model.save_pretrained("fine_tuned_llama3_8b")
tokenizer.save_pretrained("fine_tuned_llama3_8b")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("fine_tuned_llama3_8b")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_llama3_8b")

def generate_term_and_definition(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    
    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Example usage
prompt = "Generate a business term and definition for a column representing the total price of an order."
generated_result = generate_term_and_definition(prompt)
print(f"Generated Business Term and Definition: {generated_result}")
