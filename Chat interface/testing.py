import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load preprocessed dataset
df = pd.read_csv('preprocessed_dataset.csv')  # Make sure to provide the correct path

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

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    # Calculate average training loss
    avg_train_loss = total_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_predictions = []
    val_true_labels = []
    for batch in val_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        val_predictions.extend(predictions.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save the trained model
model_path = "bert_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")

# Evaluate the model
print("Evaluation on validation set:")
print(classification_report(val_true_labels, val_predictions))

# Print overall accuracy
overall_accuracy = accuracy_score(val_true_labels, val_predictions)
print(f"Overall Accuracy: {overall_accuracy:.4f}")
