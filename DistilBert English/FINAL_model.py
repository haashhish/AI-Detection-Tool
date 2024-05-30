# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertConfig, DistilBertForSequenceClassification
import torch.optim as optim

torch.cuda.empty_cache()

# Load the training and testing datasets
train_essays = pd.read_csv("training_dataset.csv")

# Explore the training data
train_essays.info()
train_essays.head()

# Check for class balance
sns.countplot(data=train_essays, x='generated')
plt.show()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean the text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    words = text.split()  # Tokenize
    words = [word.lower() for word in words if word.isalpha()]  # Lowercase and remove non-alphabetic words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

train_essays['clean_text'] = train_essays['text'].apply(clean_text)

# Load the training, validation, and testing datasets from CSV files
training_data = pd.read_csv("training_dataset.csv")
validation_data = pd.read_csv("validation_dataset.csv")
testing_data = pd.read_csv("testing_dataset.csv")

print("Number of rows in training dataset:", len(training_data))
print("Number of rows in validation dataset:", len(validation_data))
print("Number of rows in testing dataset:", len(testing_data))

X_train = training_data['text']
X_valid = validation_data['text']
y_train = training_data['generated']
y_valid = validation_data['generated']

# Convert texts to format suitable for model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
encoded_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
encoded_val = tokenizer(X_valid.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)

# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_valid.values)

# Create TensorDatasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)

# DataLoader for efficient processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the configuration from DistilBert
config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)

# Load the model with updated configuration
model = DistilBertForSequenceClassification(config)
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 30

# Initialize lists to track losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    print(f"Starting Epoch {epoch + 1}/{epochs}")
    
    # Reset metrics for each epoch
    total_loss = 0
    correct_predictions_train = 0
    total_predictions_train = 0
    model.train()
        
    # Training phase
    print("\nTRAINING IS STARTING NOW")
    for batch_idx, batch in enumerate(train_loader):

        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions_train += (predictions == labels).sum().item()
        total_predictions_train += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_predictions_train / total_predictions_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print("TRAINING IS ENDING NOW")

    # Validation phase
    print("VALIDATION IS STARTING NOW")
    model.eval()
    val_loss = 0
    correct_predictions_val = 0
    total_predictions_val = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions_val += (predictions == labels).sum().item()
            total_predictions_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_predictions_val / total_predictions_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"End of Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print("VALIDATION IS ENDING NOW")

# Plotting loss and accuracy curves
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Training/Validation Loss Curves Accuracy Curves - Distil Bert Base Uncased')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training/Validation Loss Curves - Distil Bert Base Uncased')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model weights
model_path = "distilbert_model_30Epochs"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model weights saved at: {model_path}")