from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# List of reference strings
reference_strings = [
    "customer full name",
    "lawyer full name",
    "user email address"
]

def predict_masked_text(input_string, reference_strings):
    # Tokenize input string
    tokens = tokenizer.tokenize(input_string)
    input_ids = tokenizer.encode(input_string, return_tensors='pt')
    
    best_match = None
    best_score = 0
    
    # Iterate over each token position
    for i in range(len(tokens)):
        if tokens[i] == '[MASK]':  # Skip if already masked
            continue
        
        # Prepare masked input
        masked_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
        masked_input = tokenizer.convert_tokens_to_string(masked_tokens)
        
        # Tokenize masked input and get model outputs
        masked_input_ids = tokenizer.encode(masked_input, return_tensors='pt')
        with torch.no_grad():
            outputs = model(masked_input_ids)
        
        # Get the predicted token probabilities for [MASK]
        predictions = outputs.logits[0, tokenizer.convert_tokens_to_ids('[MASK]')]

        # Decode predictions to find top tokens
        predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.topk(5).indices.tolist())
        
        # Replace [MASK] with predicted tokens and match with reference strings
        for token in predicted_tokens:
            pred_sentence = masked_input.replace('[MASK]', token)
            match_score = compare_strings(pred_sentence, reference_strings)
            if match_score > best_score:
                best_score = match_score
                best_match = pred_sentence

    return best_match

def compare_strings(pred_sentence, reference_strings):
    """Simple comparison function; can be improved with more sophisticated measures."""
    best_match = None
    best_score = 0
    for ref in reference_strings:
        # Basic string comparison, can be enhanced with similarity measures
        if pred_sentence.lower() == ref.lower():
            best_score = 1
            best_match = ref
    return best_score

# Example usage
input_string = "customer name"
best_match = predict_masked_text(input_string, reference_strings)
print(f"Best match: {best_match}")
