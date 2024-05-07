import joblib
import numpy as np

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the model
loaded_model = joblib.load('svm_model.pkl')

# Define sample test data
X_test = ["shark", "pen", "email"]

# Preprocess test data using the TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Use the vocabulary of the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Initialize a list to store predictions
predictions = []

# Loop through the test data features
for i, data in enumerate(X_test):
    # Check if the feature exactly matches any feature in the vocabulary
    if data in feature_names:
        # If it's an exact match, preprocess the data and make a prediction
        feature_index = np.where(feature_names == data)[0][0]
        print(data)
        # Create a dummy feature vector with all zeros except for the matched feature
        dummy_feature_vector = np.zeros((X_test_tfidf.shape[0], len(feature_names)))
        dummy_feature_vector[i, feature_index] = 1  # Set the matched feature to 1
        # Make a prediction using the dummy feature vector
        prediction = loaded_model.predict(dummy_feature_vector)

        predictions.append(prediction[0])
    else:
        # If there's no exact match, append an empty string to indicate no prediction
        predictions.append("")

# Print the predictions
for data, prediction in zip(X_test, predictions):
    print(f"Data: {data}, Predicted Label: {prediction}")
