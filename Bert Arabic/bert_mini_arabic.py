# Import necessary libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from transformers import BertConfig
import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoConfig, AutoModelForSequenceClassification
import plotly.graph_objects as go

nltk.download('stopwords')
torch.cuda.empty_cache()

def clean_text(text):
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize Arabic text
    words = text.split()

    # Load Arabic stop words
    arabic_stop_words = set(stopwords.words('arabic'))

    # Filter non-stop words and normalize Arabic text
    clean_words = []
    for word in words:
        # Remove diacritics from Arabic letters
        word = ''.join(c for c in word if not c in ('َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ', 'ـ'))
        # Remove non-alphabetic characters
        if word.isalpha():
            # Lowercase and remove Arabic stop words
            if word.lower() not in arabic_stop_words:
                clean_words.append(word.lower())

    # Join cleaned words back to text
    clean_text = ' '.join(clean_words)
    return clean_text

def clean_text_2(text):
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # Remove punctuation and non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    # Normalize the text
    text = unicodedata.normalize('NFKD', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    cleaned_text = clean_text_2(text)
    #return tokenizer(cleaned_text, padding=True, truncation=True, return_tensors='pt')
    return cleaned_text

def predict_realtime_with_threshold(text, threshold=0.5):
    #Preprocess the text
    cleaned_text = preprocess_text(text)
    input_data = tokenizer(cleaned_text, padding=True, truncation=True, return_tensors='pt', max_length=512)

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

train_essays = pd.read_csv("Arabic_training_dataset.csv", encoding='utf-8')
validation_essays = pd.read_csv("Arabic_validation_dataset.csv", encoding='utf-8')
testing_essays = pd.read_csv("final3400HumanizedArabic.csv", encoding='utf-8')

# train_essays = pd.read_csv("ArabicPH_train_set.csv", encoding='utf-8')
# validation_essays = pd.read_csv("ArabicPH_valid_set.csv", encoding='utf-8')
# testing_essays = pd.read_csv("ParaphrasedArabic3824.csv", encoding='utf-8')

# train_essays = pd.read_csv("ArabicPH_train_set.csv", encoding='utf-8')
# validation_essays = pd.read_csv("ArabicPH_valid_set.csv", encoding='utf-8')
# testing_essays = pd.read_csv("final3400HumanizedArabic.csv", encoding='utf-8')

# train_essays = pd.read_csv("PHFinalTraining_Arabic.csv", encoding='utf-8')
# validation_essays = pd.read_csv("PHFinalValidation_Arabic.csv", encoding='utf-8')
# # testing_essays = pd.read_csv("Arabic_testing_dataset.csv", encoding='utf-8')
# testing_essays = pd.read_csv("ParaphrasedArabic3824.csv", encoding='utf-8')

# Explore the training data
train_essays.info()
train_essays.head()

# # Check for class balance
# sns.countplot(data=train_essays, x='generated')
# plt.show()

for index in train_essays.index:
    essay = train_essays.at[index, 'text']
    try:
        cleaned_essay = clean_text_2(essay)
        train_essays.at[index, 'text'] = cleaned_essay
    except Exception as e:
        print(f"Error processing essay at index {index}: {e}")
        continue

train_essays['text'] = train_essays['text'].apply(clean_text_2)
train_essays['text'] = train_essays.iloc[1:]['text'].apply(clean_text_2)

X_train = train_essays['text']
X_valid = validation_essays['text']
X_test = testing_essays['text']
y_train = train_essays['generated']
y_valid = validation_essays['generated']
y_test = testing_essays['generated']

# Print the sizes of each set
print("Training set size:", len(X_train))
print("Validation set size:", len(X_valid))
print("Testing set size:", len(X_test))

# Ensure that X_train is a list of strings (which you've already done)
X_train = [str(x) for x in X_train]
X_valid = X_valid.astype(str)

#max_length=512
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-mini-arabic", padding=True, truncation=True, max_length=256)

# Tokenization and Encoding for BERT
encoded_train = tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors='pt')
encoded_val = tokenizer([str(x) for x in X_valid], padding=True, truncation=True, max_length=256, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_valid.values)

# Create TensorDatasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)

# DataLoader for efficient processing
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) #originally 16
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

config = AutoConfig.from_pretrained('asafaya/bert-mini-arabic', num_labels=2)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3

model = AutoModelForSequenceClassification.from_pretrained('asafaya/bert-mini-arabic', config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device} for training")
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=True)
epochs = 30 #10 originally

# print("\nTRAINING IS STARTING NOW")
# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []

# for epoch in range(epochs):
#     print(f"EPOCH NUMBER: {epoch + 1}")

#     total_loss = 0
#     correct_train_predictions = 0
#     total_train_samples = 0

#     # Set the model to training mode
#     model.train()

#     for batch_idx, batch in enumerate(train_loader):

#         input_ids, attention_mask, labels = batch
#         input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

#         # Zero the gradients before running the forward pass.
#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
    
#         # l2_reg_loss = sum(torch.norm(param) ** 2 for param in model.parameters())
#         # loss += 0.5 * 0.01 * l2_reg_loss #best 0.02
    
#         # Calculate training accuracy
#         total_loss += loss.item()
#         logits = outputs.logits
#         _, predicted = torch.max(logits, 1)
#         correct_train_predictions += (predicted == labels).sum().item()
#         total_train_samples += labels.size(0)

#         # Backward pass and optimize
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
#         optimizer.step()

#         if (batch_idx + 1) % 100 == 0:
#             print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

#     avg_train_loss = total_loss / len(train_loader)
#     train_accuracy = correct_train_predictions / total_train_samples
#     train_losses.append(avg_train_loss)
#     train_accuracies.append(train_accuracy)

#     print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.2f}, Training Accuracy: {train_accuracy:.2f}")

#     # Validation phase
#     model.eval()
#     total_val_loss = 0
#     correct_val_predictions = 0
#     total_val_samples = 0

#     with torch.no_grad():
#         for val_batch in val_loader:
#             val_input_ids, val_attention_mask, val_labels = val_batch
#             val_input_ids, val_attention_mask, val_labels = val_input_ids.to(device), val_attention_mask.to(device), val_labels.to(device)

#             val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
#             val_loss = val_outputs.loss
#             total_val_loss += val_loss.item()
        
#             # Calculate validation accuracy
#             val_logits = val_outputs.logits
#             _, val_predicted = torch.max(val_logits, 1)
#             correct_val_predictions += (val_predicted == val_labels).sum().item()
#             total_val_samples += val_labels.size(0)

#     avg_val_loss = total_val_loss / len(val_loader)
#     val_accuracy = correct_val_predictions / total_val_samples
#     val_losses.append(avg_val_loss)
#     val_accuracies.append(val_accuracy)

#     print("Total Validation Loss:", total_val_loss)
#     print("Number of Validation Batches:", len(val_loader))
#     print("Total Validation Samples:", total_val_samples)

#     print(f"Epoch {epoch + 1}/{epochs}, Average Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")
#     print("Train Losses:", train_losses)
#     print("Train Accuracies:", train_accuracies)
#     print("Validation Losses:", val_losses)
#     print("Validation Accuracies:", val_accuracies)

# print("TRAINING IS ENDING NOW")

# # # #Plotting training and validation loss
# # # plt.figure(figsize=(10, 5))
# # # plt.plot(train_losses, label='Training Loss')
# # # plt.plot(val_losses, label='Validation Loss')
# # # plt.title('Training and Validation Loss')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Loss')
# # # plt.legend()
# # # plt.show()

# # # #Plotting training and validation accuracy
# # # plt.figure(figsize=(10, 5))
# # # plt.plot(train_accuracies, label='Training Accuracy')
# # # plt.plot(val_accuracies, label='Validation Accuracy')
# # # plt.title('Training and Validation Accuracy')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Accuracy')
# # # plt.legend()
# # # plt.show()

# # plt.figure(figsize=(12, 6))
# # plt.subplot(1, 2, 1)
# # plt.plot(train_losses, label='Training Loss')
# # plt.plot(val_losses, label='Validation Loss')
# # plt.title('Loss Curves')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.subplot(1, 2, 2)
# # plt.plot(train_accuracies, label='Training Accuracy')
# # plt.plot(val_accuracies, label='Validation Accuracy')
# # plt.title('Accuracy Curves')
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# epochs = range(1, epochs + 1)
# # Create traces for training and validation loss
# trace1 = go.Scatter(
#     x=list(epochs),
#     y=train_losses,
#     mode='lines+markers',
#     name='Training Loss',
#     marker=dict(color='blue')
# )

# trace2 = go.Scatter(
#     x=list(epochs),
#     y=val_losses,
#     mode='lines+markers',
#     name='Validation Loss',
#     marker=dict(color='red')
# )

# # Create traces for training and validation accuracy
# trace3 = go.Scatter(
#     x=list(epochs),
#     y=train_accuracies,
#     mode='lines+markers',
#     name='Training Accuracy',
#     marker=dict(color='purple')
# )

# trace4 = go.Scatter(
#     x=list(epochs),
#     y=val_accuracies,
#     mode='lines+markers',
#     name='Validation Accuracy',
#     marker=dict(color='green')
# )


# # Create the figure and add traces for loss
# fig = go.Figure()
# fig.add_trace(trace1)
# fig.add_trace(trace2)

# # Set layout for loss plot
# fig.update_layout(
#     title='Training/Validation Loss - Bert Mini Arabic',
#     xaxis_title='Epoch',
#     yaxis_title='Loss',
#     legend_title='Legend',
#     width=600,  # adjust size as needed
#     height=400
# )

# # Show the figure
# fig.show()

# # Create a new figure for accuracy
# fig2_accuracy = go.Figure()
# fig2_accuracy.add_trace(trace3)
# fig2_accuracy.add_trace(trace4)

# # Set layout for accuracy plot
# fig2_accuracy.update_layout(
#     title='Training/Validation Accuracy - Bert Mini Arabic',
#     xaxis_title='Epoch',
#     yaxis_title='Accuracy',
#     legend_title='Legend',
#     width=600,  # adjust size as needed
#     height=400
# )

# # Show the figure
# fig2_accuracy.show()

#Save the model weights
model_path = "bert_mini_Arabic_30Epochs"
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)
# print(f"Model and tokenizer weights saved at: {model_path}")

print('-----------------------------------------\n')

# Load the saved model and tokenizer
loaded_model = BertForSequenceClassification.from_pretrained(model_path)
loaded_model = loaded_model.to(device)
# Move model to the same device
loaded_tokenizer = BertTokenizer.from_pretrained(model_path)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for essay_text, label in zip(X_test, y_test):
    predicted_class, predicted_prob = predict_realtime_with_threshold(essay_text)
    print(f"Actual Class: {label}, Predicted Class: {predicted_class}")
    
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

# import sys
# print(sys.version)
# print(sys.executable)

#batch_size = 4 increase it to reduce underfitting
# config.hidden_dropout_prob = 0.5
# config.attention_probs_dropout_prob = 0.5
#weight_decay=0.05 reduce it to reduce underfitting

# l2_reg_loss = sum(torch.norm(param) ** 2 for param in model.parameters())
# loss += 0.5 * 0.01 * l2_reg_loss #best 0.02