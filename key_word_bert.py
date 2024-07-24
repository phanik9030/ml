import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Example text
text = "Machine learning is a method of data analysis that automates analytical model building."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, is_split_into_words=True)

# Perform prediction
with torch.no_grad():
    outputs = model(**inputs).logits

# Extract predictions
predictions = torch.argmax(outputs, dim=2)

# Decode the predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
predicted_labels = predictions.squeeze().tolist()

# Identify keywords
keywords = [tokens[i] for i in range(len(tokens)) if predicted_labels[i] == 1]

# Print keywords
print("Keywords:", keywords)
