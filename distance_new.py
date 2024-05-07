import csv
import pickle
from collections import defaultdict
from statistics import mean

class WeightedLevenshteinModel:
    def __init__(self, reference_strings, costs):
        self.reference_strings = reference_strings
        self.costs = costs

    def predict_label(self, feature):
        avg_distances = defaultdict(list)
        for label, refs in self.reference_strings.items():
            for ref in refs:
                # Compute weighted Levenshtein distance
                dist = self.weighted_levenshtein(feature, ref)
                avg_distances[label].append(dist)
        min_avg_distance = min((mean(distances), label) for label, distances in avg_distances.items())
        return min_avg_distance[1]

    def weighted_levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.weighted_levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                if ('', c2) not in self.costs:
                    self.costs[('', c2)] = 1
                if (c1, '') not in self.costs:
                    self.costs[(c1, '')] = 1
                if (c1, c2) not in self.costs:
                    self.costs[(c1, c2)] = 1

                insert_cost = previous_row[j + 1] + self.costs[('', c2)]
                delete_cost = current_row[j] + self.costs[(c1, '')]
                substitute_cost = previous_row[j] + (self.costs[(c1, c2)] if c1 != c2 else 0)
                current_row.append(min(insert_cost, delete_cost, substitute_cost))

            previous_row = current_row

        return previous_row[-1]

# Read data from CSV file
def read_data_from_csv(filename):
    reference_strings = defaultdict(list)
    costs = {}

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']
            if row['feature_column'] not in reference_strings[label]:
                reference_strings[label].append(row['feature_column'])

            for char in row['feature_column']:
                if ('', char) not in costs:
                    costs[('', char)] = 1
                if (char, '') not in costs:
                    costs[(char, '')] = 1
                if ('', char) not in costs:
                    costs[('', char)] = 1

    return reference_strings, costs

# Define CSV filename
csv_filename = 'data.csv'

# Read data from CSV
reference_strings, costs = read_data_from_csv(csv_filename)

# Create an instance of the model
model = WeightedLevenshteinModel(reference_strings, costs)

# Export the model
with open('weighted_levenshtein_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from the file
with open('weighted_levenshtein_model.pkl', 'rb') as f:
    model = pickle.load(f)

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
    predicted_label = model.predict_label(feature)
    print(f"Feature: {feature}, Predicted Label: {predicted_label}, Expected Label: {expected_label}")
