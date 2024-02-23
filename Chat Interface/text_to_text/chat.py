import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load preprocessed dataset
preprocessed_dataset = pd.read_csv("preprocessed_dataset.csv")

# Load user input
user_input = input("Please describe the disease symptom: ")

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Preprocess user input
processed_user_input = preprocess_text(user_input)

# Preprocess text column from the dataset
preprocessed_dataset['processed_text'] = preprocessed_dataset['text'].apply(preprocess_text)

# Combine user input and dataset text for TF-IDF vectorization
all_text = [processed_user_input] + list(preprocessed_dataset['processed_text'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)

# Calculate cosine similarity between user input and dataset text
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Get indices of top 3 similar texts
top_indices = cosine_similarities.argsort()[-3:][::-1]

# Get corresponding labels
top_labels = preprocessed_dataset.iloc[top_indices]['label']

print("Top 3 matched disease labels:")
print(top_labels)

