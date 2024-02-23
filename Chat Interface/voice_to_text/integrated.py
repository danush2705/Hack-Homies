import speech_recognition as sr
from googletrans import Translator
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load the dataset
symptom_data = pd.read_csv("Symptom2Disease.csv")

def transcribe_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak in Tamil:")
        audio = recognizer.listen(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio, language="ta-IN")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def translate_text(text):
    translator = Translator()
    translated_text = translator.translate(text, src='ta', dest='en').text
    return translated_text

def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join(words)

def get_top_3_diseases(text):
    # Compute TF-IDF vectors for the dataset text
    vectorizer = TfidfVectorizer()
    dataset_text = symptom_data['text'].apply(preprocess_text)
    dataset_tfidf = vectorizer.fit_transform(dataset_text)

    # Preprocess and compute TF-IDF vector for the user input text
    input_text = preprocess_text(text)
    input_tfidf = vectorizer.transform([input_text])

    # Calculate cosine similarity between input and dataset
    similarities = cosine_similarity(input_tfidf, dataset_tfidf)

    # Find the indices of the top 3 most similar rows
    top_3_indices = similarities.argsort(axis=1)[0][-3:][::-1]

    # Get the corresponding disease labels
    top_3_diseases = [symptom_data.loc[index, 'label'] for index in top_3_indices]

    return top_3_diseases

def main():
    print("Tamil to English Speech-to-Text and Translation")
    translated_text = transcribe_speech()
    print("You (Tamil):", translated_text)
    
    print("Translating to English...")
    translated_text_english = translate_text(translated_text)
    print("You (English):", translated_text_english)

    # Extract keywords from the translated text
    keywords = re.findall(r'\b\w+\b', translated_text_english.lower())

    # Find the top 3 diseases for each keyword
    top_3_diseases = []
    for keyword in keywords:
        diseases_for_keyword = get_top_3_diseases(keyword)
        top_3_diseases.extend(diseases_for_keyword)

    # Count the occurrences of each disease
    disease_counts = Counter(top_3_diseases)

    # Select the top 3 most frequent diseases
    most_common_diseases = disease_counts.most_common(3)

    print("Predicted Symptoms for Disease:")
    for disease, _ in most_common_diseases:
        print(disease)

if __name__ == "__main__":
    main()
