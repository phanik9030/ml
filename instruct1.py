!pip install datasets
!pip install -U transformers peft bitsandbytes accelerate

import json
import torch
from datasets import Dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

def prepare_data(data):
    """Prepares data in Llama 3.1 Instruct chat format with user/assistant headers."""
    if not data or not isinstance(data, list):
        raise ValueError("Input data must be a non-empty list of dictionaries.")
    
    formatted_examples = []
    
    for item in data:
        if "term" in item and "definition" in item:
            user_prompt = f"What is the definition of {item['term']}?"
            assistant_response = item['definition']
        elif "column" in item and "associatedTerm" in item:
            clean_column = item["column"].replace("_", " ").lower()
            user_prompt = f"What is the business-friendly term for the column '{clean_column}'?"
            assistant_response = item['associatedTerm']
        else:
            continue  # Skip invalid entries

        conversation = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n"
            f"{assistant_response}<|eot_id>"
        )
        
        formatted_examples.append({"text": conversation})
    
    if not formatted_examples:
        raise ValueError("No valid examples found in the data.")
    
    return Dataset.from_list(formatted_examples)

# 2. Tokenization
def tokenize_function(examples, tokenizer):
    """ Tokenizes with labels for causal LM """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# In the setup_model function, after loading the tokenizer
def setup_model(model_name):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # load_in_8bit=True,  # Switch to 8-bit
        # bnb_8bit_compute_dtype=torch.float16,
        # bnb_8bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Add special tokens if they don't exist
    special_tokens = {
        "additional_special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.15,  # Increase dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer

# 4. Fine-Tuning
def fine_tune_model(data, model, tokenizer, output_dir):
    dataset = prepare_data(data)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,  # Slightly more than best epoch (8)
        per_device_train_batch_size=2,  # Increased for efficiency
        gradient_accumulation_steps=8,  # Effective batch size 16
        learning_rate=3e-5,  # Lowered for finer convergence
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.05,  # Increased to reduce overfitting
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=True,
        optim="adamw_8bit",
        # early_stopping_patience=1,  # Stop if no improvement
        # early_stopping_threshold=0.005
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    

    print("Starting model training...")
    trainer.train()
    print("Training completed!")

    output_dir = ""
    print("Merging LoRA adapters into the base model...")
    model = model.merge_and_unload()  # Merge LoRA weights into the base model
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure sync
    print(f"Saving full model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Full model and tokenizer saved to {output_dir}")


    # 5. Execution
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = ""
    file_path = ""

    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    training_data = data if isinstance(data, list) else data["terms"]

    # Setup and train
    model, tokenizer = setup_model(model_name)
    fine_tune_model(training_data, model, tokenizer, output_dir)

def load_finetuned_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate_response(model, tokenizer, user_prompt):
    """
    Generate a response for a given user prompt using the fine-tuned model.
    """
    # Format the prompt in the same way as training data
    formatted_prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n"
    )

    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Further reduce to force conciseness
            num_return_sequences=1,
            temperature=0.01,  # Very low for deterministic output
            top_k=1,  # Greedy decoding
            top_p=0.9,
            do_sample=False,  # No sampling, use greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start = output_text.rfind("<|start_header_id|>assistant<|end_header_id>\n\n") + len("<|start_header_id|>assistant<|end_header_id>\n\n")
    assistant_response = output_text[assistant_start:].split("<|eot_id>")[0].strip()
    
    # Ensure no extra text
    if "<|start_header_id|>" in assistant_response or "<|end_header_id|>" in assistant_response:
        assistant_response = assistant_response.split("<|start_header_id|>")[0].strip()

    return assistant_response

if __name__ == "__main__":
    model_path = ""
    print("Loading fine-tuned model...")
    model_tuned, tokenizer_tuned = load_finetuned_model(model_path)
    print("Model loaded successfully!")

def run_test_cases(model, tokenizer):
    """
    Run a set of predefined test cases to evaluate the model's inference.
    """
    test_cases = [
        {
            "user_input": "",
            "expected_output": "",
            "description": ""
        },
    ]

    for i, test in enumerate(test_cases, 1):
        # print(f"\nTest Case {i}: {test['description']}")
        print(f"User Input: {test['user_input']}")

        # Generate response
        response = generate_response(model, tokenizer, test['user_input'])
        print(f"Model Output: {response}")

        # Check if the response matches the expected output (case-insensitive)
        is_correct = test['expected_output'].lower() in response.lower()
        print(f"Expected Output: {test['expected_output']}")
        print(f"Pass: {is_correct}")

run_test_cases(model_tuned, tokenizer_tuned)
