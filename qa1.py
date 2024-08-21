from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Prepare the input sentence
input_sentence = "The capital of France is Paris."
input_text = "generate question: " + input_sentence
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate the question
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Question:", question)

