import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove the word 'reuters'
    text = re.sub(r'\breuters\b', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Handle negation
    negation_tokens = []
    negate = False
    for token in tokens:
        if token in ["not", "n't"]:
            negate = True
            negation_tokens.append(token)
        elif negate:
            negation_tokens.append("NOT_" + token)
            negate = False
        else:
            negation_tokens.append(token)
    tokens = negation_tokens

    # Remove non-alphabetic characters
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# Load the data
input_csv = '../../data/news_data/raw/synthetic/fs2_n3000.csv'
output_csv = '../../data/news_data/preprocessed/synthetic/processed_fs2_n3000.csv'
df = pd.read_csv(input_csv)

# Combine 'subject' and 'text' columns
df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Apply preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Keep only 'text' and 'label' columns
df = df[['text', 'label']]

# Save the processed data to a new CSV file
df.to_csv(output_csv, index=False)

print(f"Preprocessing complete. Processed data saved to {output_csv}.")
