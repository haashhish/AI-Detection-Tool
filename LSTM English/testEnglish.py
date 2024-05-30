import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to clean text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Load the model
# model = load_model('lstm_English')
model = load_model('lstm_English_30Epochs')

# Load the tokenizer configuration
with open("lstmEnglishTokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_config = f.read()
tokenizer = tokenizer_from_json(tokenizer_config)

# Load your test data
test_data = pd.read_csv("testing_dataset.csv")
# test_data = pd.read_csv("final28000Paraphrased.csv")
# test_data = pd.read_csv("final28000Humanized.csv")

# test_data = pd.read_csv("English_360000.csv")
X_test = test_data['text'].apply(clean_text)

# Convert text to sequences
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=128, padding='post', truncating='post')  # Ensure this matches training maxlen

# Make predictions
predictions = model.predict(X_test_padded)

#Convert predictions to binary outcomes
predicted_classes = (predictions > 0.5).astype(int)

# Print or save the predictions
print("Predictions:", predicted_classes.flatten())

# We use the true labels that we have in order to evaluate the model
if 'generated' in test_data.columns:
    y_test = test_data['generated']
    #y_test = test_data[4]
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predicted_classes)

    # True Positives, False Positives, True Negatives, False Negatives
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision
    precision = TP / (TP + FP)

    # Recall
    recall = TP / (TP + FN)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'True Positives: {TP}')
    print(f'True Negatives: {TN}')
    print(f'False Positives: {FP}')
    print(f'False Negatives: {FN}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')
    
    # Printing the classification report for a detailed overview
    print(classification_report(y_test, predicted_classes))
