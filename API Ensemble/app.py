from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, BertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from langdetect import detect
import re
import unicodedata
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar
import os
import subprocess
import pathlib
import textwrap
import pandas as pd
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import nltk
from nltk.corpus import stopwords
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

genai.configure(api_key="API")
Geminimodel = genai.GenerativeModel('gemini-pro')

nltk.download('punkt')


#Arabic Ensemble (bert-mini and LSTM)

async def load_model_AR():
    
    bert_model = AutoModelForSequenceClassification.from_pretrained('asafaya/bert-mini-arabic').to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-mini-arabic')
    bert_model.load_state_dict(torch.load("Weights/bert_mini_arabic_model_weights.pth"))


    lstm_model = tf.keras.models.load_model('Weights/lstm_Arabic_test')
    with open('Weights/lstmArabicTokenizer_test.json', 'r', encoding='utf-8') as f:
        lstm_tokenizer_json = f.read()
    lstm_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(lstm_tokenizer_json)
    
    return bert_model, bert_tokenizer, lstm_model, lstm_tokenizer


async def clean_text_bert(text):

    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    arabic_stop_words = set(stopwords.words('arabic'))

   
    clean_words = []
    for word in words:
        word = ''.join(c for c in word if not c in ('َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ', 'ـ'))
        if word.isalpha():
            if word.lower() not in arabic_stop_words:
                clean_words.append(word.lower())

    clean_text = ' '.join(clean_words)
    return clean_text

# async def preprocess_text_multilingual(text):
#     cleaned_text = await clean_text_bert(text)
#     #return tokenizer(cleaned_text, padding=True, truncation=True, return_tensors='pt')
#     return cleaned_text


async def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'\s+', ' ', text).strip()
    elif isinstance(text, list):
        text = [await clean_text(t) for t in text]
    else:
        raise TypeError("Input must be a string or a list of strings.")
    
    return text


# def clean_text(text):
#     def clean_individual_text(t):
#         t = re.sub(r'[\u064B-\u065F\u0670]', '', t)  
#         t = re.sub(r'[^\u0600-\u06FF\s]', '', t)     
#         t = unicodedata.normalize('NFKD', t)        
#         return re.sub(r'\s+', ' ', t).strip()  

#     if isinstance(text, str):
#         return clean_individual_text(text)
#     elif isinstance(text, list):
#         return [clean_individual_text(t) for t in text]
#     else:
#         raise TypeError("Input must be a string or a list of strings.")


async def arabic_predict(text, bert_model, bert_tokenizer, lstm_model, lstm_tokenizer):

    cleaned_text = await clean_text(text)
    print("Cleaned Text bert:", cleaned_text)
    bert_inputs = bert_tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_probs = torch.softmax(bert_outputs.logits, dim=1)[:, 1].cpu().numpy() 
   
    cleaned_textt = await clean_text(text)
    print("Cleaned Text lstm:", cleaned_textt)
    lstm_seq = lstm_tokenizer.texts_to_sequences([cleaned_textt])
    print("LSTM Sequence:", lstm_seq)
    lstm_padded = tf.keras.preprocessing.sequence.pad_sequences(lstm_seq, maxlen=256, padding='post', truncating='post')
    lstm_padded = lstm_padded.reshape(1, -1) if lstm_padded.ndim == 1 else lstm_padded
    print("LSTM Padded:", lstm_padded)
    # lstm_probs = lstm_model.predict(lstm_padded)[0]
    probabilities = lstm_model.predict(lstm_padded)[:, 0]
    


    # Average predictions
    avg_prob = (bert_probs + probabilities) / 2
    # avg_prob = bert_probs
    
 
    predicted_class = "AI-generated" if avg_prob >= 0.5 else "Human-written"
    
    avg_prob_list = avg_prob.tolist() if isinstance(avg_prob, np.ndarray) else avg_prob
    return predicted_class, avg_prob_list[0]



async def identify_ai_sentences_Arabic(essay, bert_model, bert_tokenizer, lstm_model, lstm_tokenizer, device, max_length=128):
    
    ai_sentences =[]
    for sentence in essay.split('.'):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        encoding = bert_tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_ai = probs[0][1].item()
            # _, preds = torch.max(outputs, dim=1)
        if pred_ai > 0.99865:
            ai_sentences.append(sentence)
        print(len(ai_sentences), "/", len(sentence))
    return ai_sentences


#English

async def identify_ai_sentences_English(essay, distilbert_model, distilbert_tokenizer, uncased_model, uncased_tokenizer, lstm_model, lstm_tokenizer, device, max_length=128):
    ai_sentences = []
    for sentence in essay.split('.'):
        sentence = sentence.strip()
        if not sentence:
            continue

        
        dbert_encoding = distilbert_tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        dbert_input_ids = dbert_encoding['input_ids'].to(device)
        dbert_attention_mask = dbert_encoding['attention_mask'].to(device)
        with torch.no_grad():
            dbert_outputs = distilbert_model(dbert_input_ids, attention_mask=dbert_attention_mask)
            dbert_probabilities = torch.softmax(dbert_outputs.logits, dim=1)
            dbert_prob_ai_generated = dbert_probabilities[0][1].item()
            _, dbert_preds = torch.max(dbert_outputs.logits, dim=1)
        if dbert_preds.item() == 1 and dbert_prob_ai_generated > 0.997:
            ai_sentences.append(sentence)

        
        uncased_encoding = uncased_tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        uncased_input_ids = uncased_encoding['input_ids'].to(device)
        uncased_attention_mask = uncased_encoding['attention_mask'].to(device)
        with torch.no_grad():
            uncased_outputs = uncased_model(uncased_input_ids, attention_mask=uncased_attention_mask)
            uncased_probabilities = torch.softmax(uncased_outputs, dim=1)
            uncased_prob_ai_generated = uncased_probabilities[0][1].item()
            _, uncased_preds = torch.max(uncased_outputs, dim=1)
        if uncased_preds.item() == 1 and uncased_prob_ai_generated > 0.997:
            ai_sentences.append(sentence)

        inputs = await preprocess_data_for_lstm([sentence], lstm_tokenizer)
        probabilities = lstm_model.predict(inputs)[:, 0]
        
        if probabilities.item() == 1 and probabilities[0][1] > 0.997:
            ai_sentences.append(sentence)

        print(len(ai_sentences), "/", len(sentence))

    return ai_sentences


bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 256
batch_size = 8
num_epochs = 10
learning_rate = 4e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on : ",device)

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        config = BertConfig.from_pretrained(bert_model_name, attention_probs_dropout_prob=0.5)
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    

async def load_model_EN():
    bert_model_path = "Weights/bfsc_graphs"  
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
    distilbert_model.to(device)

    uncased = BERTClassifier(bert_model_name, num_classes).to(device)
    uncasedTokenizer = BertTokenizer.from_pretrained(bert_model_name)
    uncased.load_state_dict(torch.load("Weights/bert_classifier.pth"))


    lstm_model_path = 'Weights/LSTM english/lstm_English'  
    lstm_model = load_model(lstm_model_path)

    with open("Weights/LSTM english/lstmEnglishTokenizer.json", "r", encoding="utf-8") as f:
        lstm_tokenizer_json = f.read()
    lstm_tokenizer = tokenizer_from_json(lstm_tokenizer_json)
    
    return distilbert_model, distilbert_tokenizer, uncased, uncasedTokenizer, lstm_model, lstm_tokenizer
    

async def preprocess_data_for_uncased(texts, uncasedTokenizer):
    encoded = uncasedTokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    return encoded

async def preprocess_data_for_bert(texts, distilbert_tokenizer):
    encoded = distilbert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    return encoded

async def preprocess_data_for_lstm(texts, lstm_tokenizer):
    sequences = lstm_tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
    return padded_sequences

async def predict_with_bert(model, texts, distilbert_tokenizer):
    model.eval()
    with torch.no_grad():
        inputs = await preprocess_data_for_bert(texts, distilbert_tokenizer)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        probabilities = torch.nn.functional.softmax(logits, dim=1)[:, 1]  
        return probabilities.cpu().numpy()

async def predict_BBU(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_ai = probs[0][1].item()
        # _, preds = torch.max(outputs, dim=1)
    return pred_ai

async def predict_with_lstm(model, texts, lstm_tokenizer):
   
    inputs = await preprocess_data_for_lstm(texts, lstm_tokenizer)
    probabilities = model.predict(inputs)[:, 0]
    return probabilities

async def average_predictions(texts, distilbert_model, distilbert_tokenizer, uncased, uncasedTokenizer, lstm_model, lstm_tokenizer):
   
    bert_predictions = await predict_with_bert(distilbert_model, texts, distilbert_tokenizer)
    print("BERT Predictions:", bert_predictions)
    lstm_predictions = await predict_with_lstm(lstm_model, texts, lstm_tokenizer)
    print("LSTM Predictions:", lstm_predictions)
    uncased_predictions = await predict_BBU(texts, uncased, uncasedTokenizer, device)
    print("Uncased Predictions:", uncased_predictions)

    
    average_predictions = (bert_predictions + lstm_predictions + uncased_predictions) / 3
    
    
    votes = [
        "AI-generated" if bert_predictions > 0.5 else "Human-written",
        "AI-generated" if lstm_predictions > 0.5 else "Human-written",
        "AI-generated" if uncased_predictions > 0.5 else "Human-written"
    ]

    decision = max(set(votes), key=votes.count)
    
    average_predictions_list = average_predictions.tolist() if isinstance(average_predictions, np.ndarray) else average_predictions

    return decision, average_predictions_list[0]


class EssayAnalysis(BaseModel):
    essay: str
    prompt: str = None
    type: str

async def generate_English_essays(prompt: str):
    ai_generated_list = []
    for counter in range(5):
        try:
            prompt2 = f"You are a helpful assistant, can you write for me an essay of 700 words on the following topic: {prompt}"
            generated_essay = Geminimodel.generate_content(prompt2).text
            ai_generated_list.append(generated_essay)
        except Exception as e:
            print(f"Error occurred: {e}")
        await asyncio.sleep(10)
    return ai_generated_list

async def generate_English_prompt(essay: str):
    prompt = f"You are a helpful assistant, can you tell me the prompt of this essay: {essay}"
    generated_prompt = Geminimodel.generate_content(prompt).text
    return generated_prompt

def preprocess_text(text: str):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)



async def generate_Arabic_essays(prom):
    ai_generated_list = []
    counter = 0
    while counter < 5:
        try:
            prompt2 = f" please do NOT ADD '*'انت مساعد مفيد هل يمكنك انت تكتب لي مقالة من 700 كلمة عن هذا الموضوع من غير اضافة *\n{prom}"
            generated_essay = Geminimodel.generate_content(prompt2).text
            print(generated_essay)
            ai_generated_list.append(generated_essay)
            counter += 1
        except Exception as e:
            print(f"Error occurred in one of the iterations: {e}")
            counter += 1
        time.sleep(10)
    return ai_generated_list

async def generate_Arabic_prompt(ess):
    prompt=f"أنت مساعد مفيد، هل يمكنك أن تخبرني بموضوع هذه المقالة من غير ان تضيف ال*\n{ess}"
    generated_prompt = Geminimodel.generate_content(prompt).text
    print(generated_prompt )
    return generated_prompt 


def get_highest_similarity(essays, input_essay):
    preprocessed_essays = [preprocess_text(essay) for essay in essays]
    preprocessed_input_essay = preprocess_text(input_essay)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_input_essay] + preprocessed_essays)
    similarities = cosine_similarity(vectors[0], vectors[1:])
    return max(similarities[0])

@app.on_event("startup")
async def startup_event():
    await load_model_EN()
    await load_model_AR()


@app.post('/')
async def analyze_essay(request: EssayAnalysis):

    if not request.essay:
        raise HTTPException(status_code=400, detail="Please provide an essay.")
    
    language = detect(request.essay)
    if language == "en":
        distilbert_model, distilbert_tokenizer, uncased, uncasedTokenizer, lstm_model, lstm_tokenizer = await load_model_EN()
        prediction, score = await average_predictions([request.essay], distilbert_model, distilbert_tokenizer, uncased, uncasedTokenizer, lstm_model, lstm_tokenizer)
        if score*100 > 1:    
            ai_sentences = await identify_ai_sentences_English(request.essay, distilbert_model, distilbert_tokenizer, uncased, uncasedTokenizer, lstm_model, lstm_tokenizer, device)
        else:
            ai_sentences = []
        if request.type == "advanced":
            essays = await generate_English_essays(request.prompt) if request.prompt else await generate_English_essays(await generate_English_prompt(request.essay))
            cosine_similarity = get_highest_similarity(essays, request.essay)
            
            if prediction == "AI-generated":
                if score > 0.8:
                    percentage = score  
                    decision = "AI-generated"
                elif cosine_similarity > 0.8:
                    percentage = score * 0.8 + cosine_similarity * 0.2  
                    decision = "AI-generated"
                else:
                    percentage = score  
                    decision = "AI-generated"
                    
            else:  
                if score < 0.2:
                    percentage = score  
                    decision = "Human-written"
                elif cosine_similarity < 0.2:
                    percentage = score * 0.8 + cosine_similarity * 0.2
                    decision = "Human-written"
                else:
                    percentage = score
                    decision = "Human-written"
        elif request.type == "basic":
            if prediction == "AI-generated":
                decision = "AI-generated"
                percentage = score
            else:
                decision = "Human-written"
                percentage = score    
    elif language == "ar":
        bert_model, bert_tokenizer, lstm_model, lstm_tokenizer = await load_model_AR()
        prediction, score = await arabic_predict(request.essay, bert_model, bert_tokenizer, lstm_model, lstm_tokenizer)
        if score*100 > 1:
            ai_sentences = await identify_ai_sentences_Arabic(request.essay, bert_model, bert_tokenizer, lstm_model, lstm_tokenizer, device)
        else:
            ai_sentences = []
        if request.type == "advanced":
            essays = await generate_Arabic_essays(request.prompt) if request.prompt else await generate_Arabic_essays(await generate_Arabic_prompt(request.essay))
            cosine_similarity = get_highest_similarity(essays, request.essay)
            
            if prediction == "AI-generated":
                if score > 0.8:
                    percentage = score  
                    decision = "AI-generated"
                elif cosine_similarity > 0.8:
                    percentage = score * 0.8 + cosine_similarity * 0.2  
                    decision = "AI-generated"
                else:
                    percentage = score  
                    decision = "AI-generated"
                    
            else:  
                if score < 0.2:
                    percentage = score  
                    decision = "Human-written"
                elif cosine_similarity < 0.2:
                    percentage = score * 0.8 + cosine_similarity * 0.2  
                    decision = "Human-written"
                else:
                    percentage = score
                    decision = "Human-written"
                
        elif request.type == "basic":
            if prediction == "AI-generated":
                decision = "AI-generated"
                percentage = score
            else:
                decision = "Human-written"
                percentage = score



    return JSONResponse(content={
        "decision": decision,
        "percentage": percentage,
        "ai_sentences": ai_sentences
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


