import numpy as np
import pickle as pkl
import streamlit as st
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartTokenizer, BartForConditionalGeneration

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    le = pkl.load(file)

max_length= 15
classifier = load_model('News_Classification.keras')
model_name = "facebook/bart-large-cnn"
# vocab_size= 20000

def preprocess_headline(text, tokenizer, max_len=max_length):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokens, maxlen=max_len, padding='post', truncating='post')
    return padded

def decode_headline(sequence, tokenizer):
    reversed_index = {} 
    for key, index in tokenizer.word_index:
        reversed_index[key] = index
    words = []
    for i in sequence:
        word = reversed_index[i]
        words.append(word)
    return " ".join(words)


def classify(data):
    data = preprocess_headline(data, tokenizer, max_length)
    predicticed = classifier.predict(data)
    predicted = np.argmax(predicticed, axis=1)
    final = le.inverse_transform(predicted)
    return final

def summarize(article, model_name=model_name):
    barttokenizer = BartTokenizer.from_pretrained(model_name)
    bart = BartForConditionalGeneration.from_pretrained(model_name)

    tokenized = barttokenizer(article, max_length=1024, return_tensors='pt',truncation=True)
    summary = bart.generate(tokenized['input_ids'])
    summary = barttokenizer.decode(summary[0], skip_special_tokens=True)
    return  summary


st.title('News Classifier and Summarizer')
headline = st.text_input("Headline")
article = st.text_area("Article")
btn = st.button("Enter")

if btn:
    category = classify(headline)
    summary = summarize(article)
    st.text("Category:")
    st.text(category)
    st.text("summary:")
    st.text(summary)
