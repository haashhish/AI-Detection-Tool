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

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Data loading
train_essays = pd.read_csv("training_dataset.csv")
validation_data = pd.read_csv("validation_dataset.csv")
testing_data = pd.read_csv("testing_dataset.csv")

# train_essays = pd.read_csv("pH_train_set.csv")
# validation_data = pd.read_csv("pH_valid_set.csv")
# testing_data = pd.read_csv("final2800Paraphrased.csv")
# testing_data = pd.read_csv("final2800Humanized.csv")

print("Number of rows in training dataset:", len(train_essays))
print("Number of rows in validation dataset:", len(validation_data))
print("Number of rows in testing dataset:", len(testing_data))

X_train = train_essays['text']
X_val = validation_data['text']
y_train = train_essays['generated']
y_val = validation_data['generated']

sns.countplot(data=train_essays, x='generated')
plt.show()

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    words = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.isalpha() and word not in stop_words]
    return ' '.join(words)

train_essays['clean_text'] = train_essays['text'].apply(clean_text)

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=128, padding='post', truncating='post')
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train)).shuffle(1000).batch(8)

X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_val_padded = pad_sequences(X_val_sequences, maxlen=128, padding='post', truncating='post')
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_padded, y_val)).batch(8)

# Bidirectional LSTM Model with Regularization
def build_model():
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    print("Vocabulary size:", vocab_size)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, kernel_regularizer=l2(0.001))),
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
    
    model.compile(optimizer='adam',
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
plt.title('Training vs Validation Accuracy - LSTM English')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss - LSTM English')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('lstm_English_30Epochs')

tokenizer_config = tokenizer.to_json()
with open("lstmEnglishTokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_config)