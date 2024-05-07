from collections import defaultdict
from statistics import mean
from pylev import levenshtein

# Read data from CSV file
data = [
    ("golden retriever", "animal"),
    ("cat", "animal"),
    ("tiger", "animal"),
    ("lion", "animal"),
    ("bottle", "thing"),
    ("mug", "thing"),
    ("cup", "thing"),
    ("pen", "thing"),
    ("phone", "thing"),
    ("name", "data"),
    ("email", "data"),
    ("phone", "data"),
    ("dob", "data"),
    ("gender", "data"),
    ("elephant", "animal"),
    ("rabbit", "animal"),
    ("hamster", "animal"),
    ("dog", "animal"),
    ("car", "thing"),
    ("bicycle", "thing"),
    ("laptop", "thing"),
    ("book", "thing"),
    ("calendar", "thing"),
    ("address", "data"),
    ("city", "data"),
    ("state", "data"),
    ("zip code", "data"),
    ("email", "data"),
    ("mouse", "animal"),
    ("fish", "animal"),
    ("bird", "animal"),
    ("snake", "animal"),
    ("chair", "thing"),
    ("table", "thing"),
    ("lamp", "thing"),
    ("notebook", "thing"),
    ("keyboard", "thing"),
    ("age", "data"),
    ("occupation", "data"),
    ("education", "data"),
    ("height", "data"),
    ("weight", "data"),
    ("giraffe", "animal"),
    ("bear", "animal"),
    ("turtle", "animal"),
    ("crab", "animal"),
    ("hat", "thing"),
    ("glasses", "thing"),
    ("watch", "thing"),
    ("bracelet", "thing"),
    ("ring", "thing")
]

# Define reference strings for each label category
reference_strings = {
    "animal": ["dog", "cat", "elephant", "rabbit", "hamster", "mouse", "fish", "bird", "snake", "giraffe", "bear", "turtle", "crab"],
    "thing": ["book", "bottle", "mug", "cup", "pen", "phone", "car", "bicycle", "laptop", "calendar", "chair", "table", "lamp", "notebook", "keyboard", "hat", "glasses", "watch", "bracelet", "ring"],
    "data": ["email", "phone", "name", "dob", "gender", "address", "city", "state", "zip code", "age", "occupation", "education", "height", "weight"]
}

# Function to predict label based on Levenshtein distance
def predict_label(feature):
    avg_distances = defaultdict(list)
    for label, refs in reference_strings.items():
        for ref in refs:
            dist = levenshtein(feature, ref)
            avg_distances[label].append(dist)
    min_avg_distance = min((mean(distances), label) for label, distances in avg_distances.items())
    return min_avg_distance[1]

# Evaluation
correct_predictions = 0
for feature, label in data:
    predicted_label = predict_label(feature)
    if predicted_label == label:
        correct_predictions += 1

accuracy = correct_predictions / len(data)
print("Accuracy:", accuracy)

# Sample feature-label pairs
sample_data = [
    ("dog", "animal"),
    ("cat", "animal"),
    ("pen", "thing"),
    ("email", "data"),
    ("book", "thing"),
    ("city", "data")
]

# Test predictions
for feature, expected_label in sample_data:
    predicted_label = predict_label(feature)
    print(f"Feature: {feature}, Predicted Label: {predicted_label}, Expected Label: {expected_label}")
