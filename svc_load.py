import joblib

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the model
loaded_model = joblib.load('svm_model.pkl')

# Prepare sample data
sample_data = ["fish", "pen", "email@example.com"]

# Preprocess sample data using the loaded TF-IDF vectorizer
sample_data_tfidf = tfidf_vectorizer.transform(sample_data)

# Use the loaded model to make predictions
predictions = loaded_model.predict(sample_data_tfidf)

# Print the predictions
for data, prediction in zip(sample_data, predictions):
    print(f"Sample Data: {data}, Predicted Category: {prediction}")
