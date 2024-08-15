import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_URL = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}'
}

def fetch(content_prompt, retries=3, delay=1):
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": content_prompt},
            {"role": "user", "content": content_prompt}
        ],
        "n": 1,
    }
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, requests.ConnectionError, requests.HTTPError) as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Attempt {attempt + 1} failed: {e}. No more retries.")
                raise

def get_random_examples(df, label, num_examples=1):
    examples = df[df['label'] == label].sample(n=num_examples)
    example_texts = examples.to_dict(orient='records')
    return example_texts

def generate_synthetic_data(df, num_samples_per_label, num_few_shot_examples, output_path):
    synthetic_data = []

    for label in [0, 1]:  # 0 for legitimate, 1 for spam
        label_type = "legitimate" if label == 0 else "spam"
        for _ in range(num_samples_per_label // 2):  # Equal number of samples per label
            combined_examples = ""
            examples = get_random_examples(df, label, num_examples=num_few_shot_examples)
            for example in examples:
                example_text = (
                    f"subject: {example['subject']}\n"
                    f"message: {example['message']}\n"
                )
                combined_examples += example_text + "\n"

            context_prompt = (
                "You are a personal assistant designed to generate synthetic data. "
                "Now you are tasked with generating examples of {label_type} emails. "
                "Here are some examples of {label_type} emails:\n"
            ).format(label_type=label_type)

            generation_prompt = (
                "Now write an email as I required. You should imitate the examples I have provided, but you cannot simply modify or rewrite the examples I have given. "
                "Be creative and write a unique email. Strictly follow this structure:\n"
                "subject: [Subject of the email]\n"
                "message: [Body of the email]\n"
                "Make sure each field is clearly labeled and included in the response. Do not skip any fields."
            )

            full_prompt = context_prompt + combined_examples + generation_prompt
            response = fetch(full_prompt)
            generated_email = response['choices'][0]['message']['content']

            email_data = parse_email(generated_email, label)
            synthetic_data.append(email_data)

            # Save data incrementally
            synthetic_data_df = pd.DataFrame(synthetic_data, columns=['subject', 'message', 'label'])
            synthetic_data_df.to_csv(output_path, index=False, encoding='utf-8')

    return synthetic_data

def parse_email(email, label):
    email_dict = {'subject': '', 'message': '', 'label': label}
    lines = email.split('\n')
    current_field = None

    for line in lines:
        if line.startswith("subject:"):
            email_dict['subject'] = line[len("subject:"):].strip()
            current_field = 'subject'
        elif line.startswith("message:"):
            email_dict['message'] = line[len("message:"):].strip()
            current_field = 'message'
        elif current_field:
            email_dict[current_field] += ' ' + line.strip()

    return email_dict

# Load the original dataset
dataset_path = '../../data/spam_data/splits/scarce_train_data.csv'
df = pd.read_csv(dataset_path)

# Number of samples to generate for each label
num_samples_per_label = 2

# Number of few-shot examples to provide to the LLM
num_few_shot_examples = 2

# Output path
output_path = '../../data/spam_data/raw/synthetic/fs2_synthetic_email_dataset_n2.csv'

# Generate synthetic data
synthetic_data = generate_synthetic_data(df, num_samples_per_label, num_few_shot_examples, output_path)

print(f"Synthetic data saved to '{output_path}'")
