from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model for FLAN-UL2
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2")

# Input text (context from which to generate a question)
text = "The capital of France is Paris."

# Prepare the input for the model
input_text = "Generate a question: " + text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a question
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

# Decode the generated question
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Question:", question)
