
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Tokenize and encode the text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)

# Extract attention weights from the last layer
attention = outputs.attentions[-1].squeeze(0)  # shape: (num_heads, seq_len, seq_len)

# Average attention weights across all heads
attention = attention.mean(dim=0)  # shape: (seq_len, seq_len)

# Sum attention weights for each token
token_importance = attention.sum(dim=0).tolist()

# Decode tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

# Pair tokens with their importance scores
token_scores = list(zip(tokens, token_importance))

# Sort tokens by importance
sorted_tokens = sorted(token_scores, key=lambda x: x[1], reverse=True)

# Extract keywords (excluding special tokens like [CLS] and [SEP])
keywords = [token for token, score in sorted_tokens if token not in ["[CLS]", "[SEP]"]]

print("Keywords (BERT):", keywords[:10])  # Print top 10 keywords
