import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
import unicodedata
import re
from sklearn.metrics import classification_report, confusion_matrix
import re
import unicodedata
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar

def clean_text(text):
    # Normalize Unicode characters
    text = normalize_unicode(text)
    
    # Replace Alef Maksura with Yaa
    text = normalize_alef_maksura_ar(text)
    
    # Remove diacritics (harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize different forms of Alef
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    
    # Remove punctuation and non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    
    # Normalize the text (decomposing and recomposing unicode characters)
    text = unicodedata.normalize('NFKD', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Load the model
model = load_model('lstm_Arabic_30Epochs')

# Load the tokenizer configuration
with open("lstmArabicTokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_config = f.read()
tokenizer = tokenizer_from_json(tokenizer_config)

# Load your test data
test_data = pd.read_csv("Arabic_testing_dataset.csv", encoding='utf-8')
#test_data = pd.read_csv("Arabic_ParaphrasedArabic3824.csv", encoding='utf-8')
#test_data = pd.read_csv("Arabic_final3400HumanizedArabic.csv", encoding='utf-8')

X_test = test_data['text'].apply(clean_text)

# Convert text to sequences
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=256, padding='post', truncating='post')

# Make predictions
predictions = model.predict(X_test_padded)

# Optionally, convert predictions to binary outcomes
predicted_classes = (predictions > 0.5).astype(int)

# Print or save the predictions
print("Predictions:", predicted_classes.flatten())

# We use the true labels of the testing dataset to evaluate the model
if 'generated' in test_data.columns:
    y_test = test_data['generated']
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
    
    # Printing the classification report
    print(classification_report(y_test, predicted_classes))

