import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load preprocessed training data
train_data_1 = pd.read_csv('../../data/news_data/preprocessed/real_world/processed_scarce_train_data.csv')
train_data_2 = pd.read_csv('../../data/news_data/preprocessed/synthetic/processed_fs2_n3000.csv')

# Combine the scarce train data with a subset of the synthetic data
sample_size = 2999  # Adjust this sample size as needed
sampled_train_data_2 = train_data_2.sample(n=sample_size, random_state=42)
train_data = pd.concat([train_data_1, sampled_train_data_2], ignore_index=True)

# List of test set paths
test_set_paths = [
    '../../data/news_data/preprocessed/real_world/processed_test_data_1.csv',
    '../../data/news_data/preprocessed/real_world/processed_test_data_2.csv',
    '../../data/news_data/preprocessed/real_world/processed_test_data_3.csv'
]

# Separate features and labels for training data
X_train = train_data['text']
y_train = train_data['label']

# Apply TF-IDF transformation
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=10)
dt_classifier.fit(X_train_tfidf, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=tfidf_vectorizer.get_feature_names_out(), class_names=['Real', 'Fake'], filled=True)
plt.show()

# Lists to store performance metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Function to evaluate a test set
def evaluate_test_set(test_data_path):
    test_data = pd.read_csv(test_data_path)
    X_test = test_data['text'].fillna('')  # Fill NaN values with empty strings
    y_test = test_data['label']
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = dt_classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0])  # Label '0' is fake news

    accuracies.append(accuracy)
    precisions.append(precision[0])
    recalls.append(recall[0])
    f1_scores.append(f1_score[0])

    report = classification_report(y_test, y_pred)
    print(f'Test Set: {test_data_path}')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    print('\n')

# Evaluate each test set
for test_set_path in test_set_paths:
    evaluate_test_set(test_set_path)

# Calculate and print average performance metrics
average_accuracy = sum(accuracies) / len(accuracies)
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1_score = sum(f1_scores) / len(f1_scores)

print('Average Performance:')
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Fake News Precision: {average_precision}')
print(f'Average Fake News Recall: {average_recall}')
print(f'Average Fake News F1 Score: {average_f1_score}')
