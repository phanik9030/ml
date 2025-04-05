# Import necessary libraries
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch
import numpy as np

# Dataset preparation
dataset_original = load_dataset("go_emotions")
sampled_dataset = dataset_original["train"].train_test_split(train_size=0.3, seed=42)["train"]
train_test_split = sampled_dataset.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

# Define emotion labels
emotion_labels = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "anticipation",
    5: "approval", 6: "boredom", 7: "confusion", 8: "curiosity", 9: "disapproval",
    10: "disappointment", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "joy", 17: "love", 18: "nervousness",
    19: "neutral", 20: "optimism", 21: "pride", 22: "realization", 23: "relief",
    24: "remorse", 25: "sadness", 26: "surprise", 27: "trust"
}
num_labels = len(emotion_labels)
label_map = {i: emotion_labels[i] for i in range(num_labels)}

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

# Updated Preprocessing (longer max_length)
def preprocess_function(examples):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = np.zeros((len(examples["labels"]), num_labels), dtype=np.float32)
    for i, label_list in enumerate(examples["labels"]):
        for label_idx in label_list:
            labels[i, label_idx] = 1.0
    encodings["labels"] = labels
    return encodings

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[col for col in dataset["train"].column_names if col not in ["labels"]]
)

from sklearn.metrics import f1_score
# Custom Trainer for multi-label
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float().to(model.device)  # Ensure float
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)
    f1_micro = f1_score(labels, predictions, average="micro")
    # print(f"Labels sample: {labels[:5]}")
    # print(f"Predictions sample: {predictions[:5]}")
    return {"f1_micro": f1_micro}

# Step 7: Set up training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/My Drive/multi_label_classifier_bnb",
    num_train_epochs=4,  # Increased
    per_device_train_batch_size=32,  # Larger batch
    per_device_eval_batch_size=32,
    learning_rate=5e-5,  # Lower LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=1,
    logging_steps=50,
    report_to="none",
)

trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

def debug_batch(trainer):
    batch = next(iter(trainer.get_train_dataloader()))
    print("Input IDs shape:", batch["input_ids"].shape, "dtype:", batch["input_ids"].dtype)
    print("Labels shape:", batch["labels"].shape, "dtype:", batch["labels"].dtype)
debug_batch(trainer)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load your fine-tuned model and tokenizer
model_path_trained = ""
tokenizer_trained = AutoTokenizer.from_pretrained(model_path_trained)
model_trained = AutoModelForSequenceClassification.from_pretrained(model_path_trained)

# Define emotion labels (same as training)
emotion_labels = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "anticipation",
    5: "approval", 6: "boredom", 7: "confusion", 8: "curiosity", 9: "disapproval",
    10: "disappointment", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "joy", 17: "love", 18: "nervousness",
    19: "neutral", 20: "optimism", 21: "pride", 22: "realization", 23: "relief",
    24: "remorse", 25: "sadness", 26: "surprise", 27: "trust"
}

def predict_emotions(text, threshold=0.7):
    inputs = tokenizer_trained(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_trained.to(device)
    
    with torch.no_grad():
        outputs = model_trained(**inputs)
        logits = outputs.logits
    
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    predicted_indices = np.where(probabilities >= threshold)[0]
    predicted_labels = [emotion_labels[idx] for idx in predicted_indices]
    predicted_scores = [round(float(probabilities[idx]), 4) for idx in predicted_indices]
    
    return {
        "labels": predicted_labels,
        "scores": predicted_scores,
        "raw_probs": probabilities.round(4)
    }

# Example usage
if __name__ == "__main__":
    test_texts = []
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        predictions = predict_emotions(text)
        print("Predicted emotions:", predictions["labels"])
        print("Confidence scores:", predictions["scores"])
        # print("All probabilities:", dict(zip(emotion_labels.values(), predictions["raw_probs"])))
