import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from collections import Counter

# Load preprocessed dataset
df = pd.read_csv('preprocessed_dataset.csv')  # Make sure to provide the correct path

# Automatically extract symptom keywords from the dataset
all_text = ' '.join(df['text'])

def extract_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

all_words = extract_words(all_text)
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
filtered_words = [word for word in all_words if word not in stopwords]
word_counts = Counter(filtered_words)
threshold = 5  # Adjust as needed
symptom_keywords = [word for word, count in word_counts.items() if count > threshold]

print("Symptom Keywords:", symptom_keywords)

# Split dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode class labels
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Load BERT tokenizer and encode text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Convert labels to torch.long
train_labels_encoded = torch.tensor(train_labels_encoded, dtype=torch.long)
val_labels_encoded = torch.tensor(val_labels_encoded, dtype=torch.long)

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels_encoded)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels_encoded)

# Define batch size and create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a function to predict the disease label from a given description
def predict_disease(description):
    # Tokenize the input text
    inputs = tokenizer(description, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label_id = torch.argmax(logits, dim=1).item()

    # Map the predicted label id to the disease label
    predicted_disease = label_encoder.classes_[predicted_label_id]

    return predicted_disease

# Define a function to check if the input contains symptoms-related keywords
def contains_symptoms(text):
    for keyword in symptom_keywords:
        if re.search(r'\b{}\b'.format(re.escape(keyword)), text, re.IGNORECASE):
            return True
    return False

# Define the chat interface
def chat_interface():
    print("Welcome to Disease Classifier Chat Interface!")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("Please describe the symptoms: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        elif contains_symptoms(user_input):
            predicted_disease = predict_disease(user_input)
            print("Predicted Disease:", predicted_disease)
        else:
            print("No symptoms detected. Please provide a description of symptoms.")

# Run the chat interface
if __name__ == "__main__":
    chat_interface()
