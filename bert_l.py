import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Example data
strings_to_train = ["your list of more than 100 strings here"]
word_array = ["approved date", "loaded date"]

# Label encode the strings
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(strings_to_train)

# Tokenize the strings using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(strings_to_train, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Split the data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    inputs['input_ids'], labels, test_size=0.1, random_state=42)



from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=val_inputs
)

# Train the model
trainer.train()




# Save the model and tokenizer
with open('bert_model.pkl', 'wb') as f:
    pickle.dump((model, tokenizer, label_encoder), f)



# Load the model and tokenizer
with open('bert_model.pkl', 'rb') as f:
    model, tokenizer, label_encoder = pickle.load(f)


# Function to get BERT embeddings
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# Compute embeddings for the word array
word_embeddings = [get_bert_embeddings(word, model, tokenizer) for word in word_array]

# Compute similarity between each word and the training strings
similarities = []
for word_embedding in word_embeddings:
    word_similarity = []
    for train_string in strings_to_train:
        train_embedding = get_bert_embeddings(train_string, model, tokenizer)
        similarity = torch.nn.functional.cosine_similarity(word_embedding, train_embedding)
        word_similarity.append(similarity.item())
    similarities.append(word_similarity)

# Print the similarities
for word, similarity in zip(word_array, similarities):
    print(f"Similarities for '{word}':")
    for string, sim in zip(strings_to_train, similarity):
        print(f"  {string}: {sim}")
