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
import feature_extraction
from feature_extraction import read_data_from_csv, text_features, sentiment_analysis_features, calculate_readability, pos_features, save_to_csv

nltk.download('punkt')
torch.cuda.empty_cache()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    words = text.split()  # Tokenize
    words = [word.lower() for word in words if word.isalpha()]  # Lowercase and remove non-alphabetic words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

def preprocess_text(text):
    cleaned_text = clean_text(text)
    return loaded_tokenizer(cleaned_text, padding=True, truncation=True, return_tensors='pt')

def predict_realtime_with_threshold(text, threshold=0.5):
    # Preprocess the text
    input_data = preprocess_text(text)

    # Move input tensor to the same device as the model
    input_data = {key: value.to(device) for key, value in input_data.items()}

    # Generate predictions using the loaded model
    with torch.no_grad():
        outputs = loaded_model(**input_data)
        logits = outputs.logits

    # Assuming the second column corresponds to the positive class (AI-generated)
    predicted_prob = torch.softmax(logits, dim=1)[:, 1].item()

    # Adjust classification based on the threshold
    predicted_class = 1 if predicted_prob >= threshold else 0

    return predicted_class, predicted_prob

def read_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['Topic'], df['Human_essay'], df['AI_essay']

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
X_test = testing_data['text']
y_test = testing_data['generated']

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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the configuration from DistilBert
config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)

# Adjust the dropout rates
config.attention_dropout = 0.5
config.dropout = 0.5 

# Load the model with updated configuration
model = DistilBertForSequenceClassification(config)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.05, correct_bias=True)

# Load the model and tokenizer
model_path = "distilbert_model_30Epochs"
loaded_model = DistilBertForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
loaded_model.to(device)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for essay_text, label in zip(X_test, y_test):
        predicted_class, predicted_prob = predict_realtime_with_threshold(essay_text)
        print(f"Predicted Class: {predicted_class}, Predicted Probability: {predicted_prob}, Actual Class: {label}")
    
        # Compare predicted class with actual class
        if predicted_class == label:
            if predicted_class == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if predicted_class == 1:
                false_positive += 1
            else:
                false_negative += 1
            
print(f"True Positive: {true_positive}")
print(f"True Negative: {true_negative}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}")

# Calculate metrics
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

# Print metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")