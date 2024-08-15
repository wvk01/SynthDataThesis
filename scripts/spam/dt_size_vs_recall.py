import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

# Function to evaluate a test set and return recall for the spam email label
def evaluate_test_set(test_data_path, tfidf_vectorizer, dt_classifier):
    test_data = pd.read_csv(test_data_path)
    X_test = test_data['text'].fillna('')  # Fill NaN values with empty strings
    y_test = test_data['label']
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = dt_classifier.predict(X_test_tfidf)

    # Calculate recall for the spam email label (1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    return recall

# Augmenting the scarce train dataset with synthetic dataset
train_data_1 = pd.read_csv('../../data/spam_data/preprocessed/real_world/processed_scarce_train_data.csv')
train_data_2 = pd.read_csv('../../data/spam_data/preprocessed/synthetic/processed_fs2_n2000.csv')
additional_train_data = pd.read_csv('../../data/spam_data/preprocessed/real_world/processed_additional_train_data.csv')

# Different sample sizes to test
sample_sizes = [0, 500, 1000, 1500, 2000]
recalls_synthetic = []
recalls_real_world = []

# List of test set paths
test_set_paths = [
    '../../data/spam_data/preprocessed/real_world/processed_test_data_1.csv',
    '../../data/spam_data/preprocessed/real_world/processed_test_data_2.csv',
    '../../data/spam_data/preprocessed/real_world/processed_test_data_3.csv'
]

for size in sample_sizes:
    # Combine the scarce train data with a subset of the synthetic data
    sampled_train_data_2 = train_data_2.sample(n=size, random_state=42)
    train_data_synthetic = pd.concat([train_data_1, sampled_train_data_2], ignore_index=True)

    # Combine the scarce train data with a subset of the additional real-world data
    sampled_additional_train_data = additional_train_data.sample(n=size, random_state=42)
    train_data_real_world = pd.concat([train_data_1, sampled_additional_train_data], ignore_index=True)

    # Separate features and labels for training data
    X_train_synthetic = train_data_synthetic['text'].fillna('')  # Fill NaN values with empty strings
    y_train_synthetic = train_data_synthetic['label']

    X_train_real_world = train_data_real_world['text'].fillna('')  # Fill NaN values with empty strings
    y_train_real_world = train_data_real_world['label']

    # Apply TF-IDF transformation
    tfidf_vectorizer_synthetic = TfidfVectorizer()
    X_train_tfidf_synthetic = tfidf_vectorizer_synthetic.fit_transform(X_train_synthetic)

    tfidf_vectorizer_real_world = TfidfVectorizer()
    X_train_tfidf_real_world = tfidf_vectorizer_real_world.fit_transform(X_train_real_world)

    # Train Decision Tree classifier on synthetic augmented data
    dt_classifier_synthetic = DecisionTreeClassifier(random_state=42)
    dt_classifier_synthetic.fit(X_train_tfidf_synthetic, y_train_synthetic)

    # Train Decision Tree classifier on real-world augmented data
    dt_classifier_real_world = DecisionTreeClassifier(random_state=42)
    dt_classifier_real_world.fit(X_train_tfidf_real_world, y_train_real_world)

    # Evaluate on all test sets and calculate average recall for synthetic augmented data
    test_recalls_synthetic = [evaluate_test_set(test_set_path, tfidf_vectorizer_synthetic, dt_classifier_synthetic) for test_set_path in test_set_paths]
    average_recall_synthetic = sum(test_recalls_synthetic) / len(test_recalls_synthetic)
    recalls_synthetic.append(average_recall_synthetic)

    # Evaluate on all test sets and calculate average recall for real-world augmented data
    test_recalls_real_world = [evaluate_test_set(test_set_path, tfidf_vectorizer_real_world, dt_classifier_real_world) for test_set_path in test_set_paths]
    average_recall_real_world = sum(test_recalls_real_world) / len(test_recalls_real_world)
    recalls_real_world.append(average_recall_real_world)

# Plot size vs recall for both synthetic and real-world augmented data
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, recalls_synthetic, marker='o', label='Synthetic Data Augmentation')
plt.plot(sample_sizes, recalls_real_world, marker='o', label='Real-World Data Augmentation')
plt.xlabel('Sample Size')
plt.ylabel('Average Recall for Spam Email')
plt.title('Sample Size vs Average Recall (Decision Tree)')
plt.grid(True)
plt.legend()
plt.show()
