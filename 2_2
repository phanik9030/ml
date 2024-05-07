import joblib
import numpy as np
from sklearn.metrics import accuracy_score

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the model
loaded_model = joblib.load('svm_model.pkl')

# Define sample test data
X_test = ["shark"]

# Preprocess test data using the TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Use the trained model to obtain decision function scores on the test data
decision_scores = loaded_model.decision_function(X_test_tfidf)

# Define a threshold for decision function scores
confidence_threshold = 1  # You can adjust this threshold as needed
print(decision_scores)
# Initialize a list to store modified predictions
modified_predictions = []

# Loop through the decision function scores and check if confidence is below the threshold
for decision_score in decision_scores:
    max_confidence = np.max(decision_score)
    print(max_confidence < confidence_threshold)
    if max_confidence < confidence_threshold:
        modified_predictions.append("Empty")
    else:
        max_index = np.argmax(decision_score)  # Find the index of the maximum confidence score
        modified_predictions.append(loaded_model.classes_[max_index])

print(modified_predictions)
# Calculate accuracy
# accuracy = accuracy_score(y_test, modified_predictions)

# print(f"Accuracy: {accuracy}")
