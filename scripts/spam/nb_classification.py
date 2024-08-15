import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Load preprocessed training data
train_data = pd.read_csv('../../data/spam_data/preprocessed/real_world/processed_scarce_train_data.csv')

# List of test set paths
test_set_paths = [
    '../../data/spam_data/preprocessed/real_world/processed_test_data_1.csv',
    '../../data/spam_data/preprocessed/real_world/processed_test_data_2.csv',
    '../../data/spam_data/preprocessed/real_world/processed_test_data_3.csv'
]

# Separate features and labels for training data
X_train = train_data['text']
y_train = train_data['label']

# Apply TF-IDF transformation
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train Naïve Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

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
    y_pred = nb_classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[1])  # Label '1' is spam email

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
print(f'Average Spam Email Precision: {average_precision}')
print(f'Average Spam Email Recall: {average_recall}')
print(f'Average Spam Email F1 Score: {average_f1_score}')
