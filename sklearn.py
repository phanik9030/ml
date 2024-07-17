from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample data
column_names = ["user_id", "user_name", "email_address", "created_at", 
                "customer_id", "customer_name", "contact_email", "creation_date", 
                "usr_id", "usr_name", "email", "date_created"]

# Normalize column names
def normalize_column_name(column_name):
    return column_name.lower().replace("_", "").replace("-", "")
    
normalized_column_names = [normalize_column_name(name) for name in column_names]

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(normalized_column_names)

# Clustering using K-Means
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_

# Display results
clusters = {}
for label, column_name in zip(labels, column_names):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(column_name)

for cluster, names in clusters.items():
    print(f"Cluster {cluster}: {names}")
