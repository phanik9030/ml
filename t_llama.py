import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# model_name = "meta-llama/Llama-3.2-1B-Instruct"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization and fp16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Enable 4-bit quantization
    torch_dtype=torch.float16,  # Use fp16 for mixed-precision training
    bnb_4bit_compute_dtype=torch.float16,  # Ensures stable computation in 4-bit mode
    device_map="auto",
)

# If GPU Memory Allows (Best Quality)
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "down_proj", "up_proj"      # Feedforward layers
]

# If You Face GPU OOM (Balanced Approach)
target_modules=["q_proj", "v_proj"]

# If Your Dataset Is Small (Prevent Overfitting)
target_modules=["q_proj", "v_proj", "gate_proj"]

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "down_proj", "up_proj"      # Feedforward layers
    ],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# training_data = []
def generate_input_output_pairs(data_list):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in data_list:
        if 'term' in item and 'definition' in item:
            prompt = {"role": "user", "content": f"What is the definition of {item['term']}?"}
            target_response = item['definition']
        elif 'column' in item and 'associatedTerm' in item:
            prompt = {"role": "user", "content": f"What does the column {item['column']} refer to?"}
            target_response = item['associatedTerm']
        else:
            # print(f"Skipping invalid entry: {item}")  # Debugging line
            continue  # Skips invalid entries

        # Create the prompt-response pair
        full_prompt = [
            {"role": "user", "content": prompt["content"]},
            {"role": "assistant", "content": target_response}
        ]
        
        # Tokenize without returning tensors yet
        tokenized = tokenizer.apply_chat_template(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=64)
        tokenized = tokenized.to(device)
        
        input_ids = tokenized.clone()
        labels = tokenized.clone()
        
        # Mask user input in labels
        user_end = (tokenized == tokenizer.eos_token_id).nonzero(as_tuple=False)[0, 1].item()
        labels[:, :user_end] = -100
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(torch.ones_like(tokenized))

    # Find the maximum sequence length
    max_length = max(t.size(1) for t in input_ids_list)

    # Pad all tensors to the maximum length
    padded_input_ids = torch.stack([torch.nn.functional.pad(t.squeeze(0), (0, max_length - t.size(1))) for t in input_ids_list], dim=0)
    padded_labels = torch.stack([torch.nn.functional.pad(t.squeeze(0), (0, max_length - t.size(1))) for t in labels_list], dim=0)
    padded_attention_mask = torch.stack([torch.nn.functional.pad(t.squeeze(0), (0, max_length - t.size(1)), value=0) for t in attention_mask_list], dim=0)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_mask
    }


# Generate training data
data = generate_input_output_pairs(training_data)

# print("Sample Input IDs:", data["input_ids"][0])
# print("Sample Labels:", data["labels"][0])

# Use 8-bit optimizer for memory efficiency
optimizer = AdamW8bit(model.parameters(), lr=3e-5, weight_decay=0.01)

# Enable gradient accumulation for larger datasets
gradient_accumulation_steps = 4

from torch.utils.data import DataLoader, TensorDataset

# ✅ Convert data to PyTorch dataset
dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
batch_size = 6  # ✅ Adjust this based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler()  # ✅ Mixed precision for efficiency
accumulation_steps = 6  # ✅ Adjust based on batch size & GPU memory

# ✅ Training Loop with Mini-Batch & Gradient Accumulation
total_steps = len(dataloader)  # Total number of steps per epoch
print(f"Total Steps per Epoch: {total_steps}")

for epoch in range(1):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        with torch.cuda.amp.autocast():  # ✅ Mixed Precision Training
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # ✅ Normalize loss

        scaler.scale(loss).backward()  # ✅ Scale loss for stability

        if (step + 1) % accumulation_steps == 0:  # ✅ Update weights after `accumulation_steps`
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # ✅ Free memory after updates

        total_loss += loss.item() * accumulation_steps  # Re-scale loss

        if (step + 1) % 100 == 0:
          print(f"Epoch {epoch + 1}, Step {step + 1}/{total_steps}, Loss: {loss.item() * accumulation_steps:.4f}")

    print(f"Epoch {epoch + 1} Completed, Average Loss: {total_loss / len(dataloader):.4f}")

# Test generation
model.eval()
with torch.no_grad():
    test_prompts = [
        {"role": "user", "content": "What is the definition of ?"},
        {"role": "user", "content": "What does the column refer to?"}
    ]

    for prompt in test_prompts:
        test_input = tokenizer.apply_chat_template([prompt], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        test_out = model.generate(
            test_input,
            max_new_tokens=128,
            do_sample=False,
            attention_mask=torch.ones_like(test_input),
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"Question: {prompt['content']}")
        print("Response:", tokenizer.batch_decode(test_out, skip_special_tokens=True)[0])
        print()
