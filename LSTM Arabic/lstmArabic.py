import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, BatchNormalization
import unicodedata
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GlobalMaxPool1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential

nltk.download('stopwords')
nltk.download('wordnet')

train_essays = pd.read_csv("Arabic_training_dataset.csv", encoding='utf-8')
validation_data = pd.read_csv("Arabic_validation_dataset.csv", encoding='utf-8')
testing_data = pd.read_csv("Arabic_testing_dataset.csv", encoding='utf-8')

# train_essays = pd.read_csv("LargeTraining.csv", encoding='utf-8')
# validation_data = pd.read_csv("LargeValidation.csv", encoding='utf-8')
# testing_data = pd.read_csv("LargeTesting.csv", encoding='utf-8')

# train_essays = pd.read_csv("ArabicPH_train_set.csv")
# validation_data = pd.read_csv("ArabicPH_valid_set.csv")
# testing_data = pd.read_csv("Arabic_ParaphrasedArabic3824.csv")
# testing_data = pd.read_csv("Arabic_final3400HumanizedArabic.csv", encoding='utf-8')

print("Number of rows in training dataset:", len(train_essays))
print("Number of rows in validation dataset:", len(validation_data))
print("Number of rows in testing dataset:", len(testing_data))

X_train = train_essays['text']
X_val = validation_data['text']
y_train = train_essays['label']
y_val = validation_data['label']

sns.countplot(data=train_essays, x='label')
plt.show()

def clean_text(text):
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # Remove punctuation and non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    # Normalize the text
    text = unicodedata.normalize('NFKD', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

train_essays['clean_text'] = train_essays['text'].apply(clean_text)

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=256, padding='post', truncating='post')

X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_val_padded = pad_sequences(X_val_sequences, maxlen=256, padding='post', truncating='post')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train)).shuffle(1000).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_padded, y_val)).batch(8)

# Model Building
def build_model(learning_rate=3.2e-06):
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    print("Vocabulary size:", vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=300),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, kernel_regularizer=l2(0.001))),
        # tf.keras.layers.GlobalMaxPool1D(),
        BatchNormalization(),
        Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

# Training and Validation
history = model.fit(train_dataset, validation_data=val_dataset, epochs=30)

# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - LSTM Arabic')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss - LSTM Arabic')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('lstm_Arabic_30Epochs')

# Save the tokenizer configuration
tokenizer_config = tokenizer.to_json()
with open("lstmArabicTokenizer_test.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_config)
