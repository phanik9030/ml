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
    # Prepare input with [MASK] token
    masked_input = input_string.replace("name", "[MASK]")
    
    # Tokenize and get model outputs
    input_ids = tokenizer.encode(masked_input, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get the predicted token probabilities for [MASK]
    predictions = outputs.logits[0, tokenizer.convert_tokens_to_ids('[MASK]')]

    # Decode predictions to find top tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.topk(5).indices.tolist())
    
    # Replace [MASK] with predicted tokens and match with reference strings
    predicted_sentences = [masked_input.replace('[MASK]', token) for token in predicted_tokens]
    
    # Find the best match
    best_match = None
    best_score = 0
    for ref in reference_strings:
        for pred_sentence in predicted_sentences:
            # Compare predicted sentence with reference string
            similarity = compare_strings(pred_sentence, ref)
            if similarity > best_score:
                best_score = similarity
                best_match = ref
    
    return best_match

def compare_strings(s1, s2):
    """Simple comparison function; can be improved with more sophisticated measures."""
    return s1.lower() == s2.lower()

# Example usage
input_string = "customer name"
best_match = predict_masked_text(input_string, reference_strings)
print(f"Best match: {best_match}")
