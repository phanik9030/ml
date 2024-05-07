import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("your_dataset.csv")

# Split dataset into features (X) and labels (y)
X = data['feature_column']
y = data['label_column']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the model (SVM classifier)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate the model
# print(classification_report(y_test, y_pred))

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Save the SVM model
joblib.dump(svm_model, 'svm_model.pkl')

# Prepare sample data
sample_data = ["fish", "pen", "email@example.com"]

# Preprocess sample data using TF-IDF vectorizer
sample_data_tfidf = tfidf_vectorizer.transform(sample_data)

# Use the model to make predictions
predictions = svm_model.predict(sample_data_tfidf)

# Print the predictions
for data, prediction in zip(sample_data, predictions):
    print(f"Sample Data: {data}, Predicted Category: {prediction}")
