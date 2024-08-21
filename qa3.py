from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model for GPT-2 medium
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Example input text
context = "The capital of France is Paris."

# Prepare the prompt for question generation
prompt = f"Generate a question about the following information: {context}"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate a question
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True, pad_token_id=tokenizer.eos_token_id)

# Decode the generated text
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Question:", question)
