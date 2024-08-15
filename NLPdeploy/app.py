import nltk
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
# import pickle
# import pandas as pd
# from scipy.sparse import csr_matrix

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Define preprocessing functions
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def to_low(x):
    return str(x).lower()

def remove_pun(x):
    return x.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def tokenize(x):
    return word_tokenize(str(x))

def lemmatize_sentence(sentence):
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(lemmatized_words)


csv_file='vocab.csv'

def load_emotion_mappings(csv_file):
    emotion_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        emotions = [field for field in reader.fieldnames if field != 'English Word']
        for row in reader:
            word = row['English Word'].lower()
            emotion_dict[word] = {emotion: int(row[emotion]) for emotion in emotions}
    return emotion_dict, emotions

def analyze_paragraph(tokens, emotion_dict, emotions, threshold=0.1):

    total_scores = defaultdict(int)
    word_count = 0

    for token in tokens:
        if token.lower() in emotion_dict:
            word_count += 1
            for emotion in emotions:
                total_scores[emotion] += emotion_dict[token.lower()][emotion]

    if word_count > 0:
        avg_scores = {emotion: total_scores[emotion] / word_count for emotion in emotions}
    else:
        avg_scores = {emotion: 0 for emotion in emotions}

    emotion_vector = [1 if avg_scores[emotion] > threshold else 0 for emotion in emotions]
    return emotion_vector

from flask import Flask

app = Flask(__name__)
model1=joblib.load('first_predictor.pkl')
# model2=joblib.load('second_predictor.pkl')

@app.route("/")
def hello_world():
    text = "Hello! I am very sad, i am crying"
    text = to_low(text)
    text = remove_pun(text)
    text = remove_stopwords(text)
    tokens = tokenize(text)

    print(tokens)

    # Load emotion mappings
    emotion_dict, emotions = load_emotion_mappings(csv_file)

    # Analyze the paragraph
    result_vector = analyze_paragraph(tokens, emotion_dict, emotions)

    return str(result_vector)
    # print(result_vector)
    # return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True,port=8080)


    