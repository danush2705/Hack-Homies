import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('Symptom2Disease.csv')  # Replace 'Symptom2Disease.csv' with your dataset file path

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply preprocessing to 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Optional: Remove empty rows if any
df = df.dropna()

# Save preprocessed dataset
df.to_csv('preprocessed_dataset.csv', index=False)  # Save to a new CSV file

# Print first few rows of preprocessed dataset
print(df.head())
