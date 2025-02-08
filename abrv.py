import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16 if device == "cuda" else torch.float32)

# Move the model to the GPU if available
model.to(device)

# List of words to check
words = ["Accumulated", "Accumulator", "Accumulation", "XYZ123", "mrkt"]

# Loop through each word
for word in words:
    # Define the input prompt in an instruction-following format
    prompt = (
        f"Is this word '{word}' in dictonary?"
    )

    # Tokenize the input and move to the GPU if available
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output using greedy decoding
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=False,  # Disable sampling for deterministic output
            temperature=0.0,  # Ensure greedy decoding
        )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract "Yes" or "No" from the response
    response_lower = response.lower()
    if "yes" in response_lower and "no" not in response_lower:
        answer = "Yes"
    elif "no" in response_lower and "yes" not in response_lower:
        answer = "No"
    else:
        answer = "Unclear"

    # Print the word and the answer
    print(f"Word: {word}")
    print(f"Is it a valid English word? {response}")
    print("-" * 40)

# List of words to check
abbrv = ["tst"]

# Loop through each word
for word in abbrv:
    # Define the input prompt in an instruction-following format
    prompt = (
        f"In the context of a database column name, what is the full form of the abbreviation '{word}'"
    )

    # Tokenize the input and move to the GPU if available
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output using greedy decoding
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=False,  # Disable sampling for deterministic output
            temperature=0.0,  # Ensure greedy decoding
        )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract "Yes" or "No" from the response
    response_lower = response.lower()
    if "yes" in response_lower and "no" not in response_lower:
        answer = "Yes"
    elif "no" in response_lower and "yes" not in response_lower:
        answer = "No"
    else:
        answer = "Unclear"

    # Print the word and the answer
    print(f"Word: {word}")
    print(f"Full word? {response}")
    print("-" * 40)
