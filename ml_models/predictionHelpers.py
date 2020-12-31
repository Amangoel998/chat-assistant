import nltk
import pickle
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from tensorflow import keras

stemmer = SnowballStemmer("english")

# Helper Functions
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

words = []
with open('./words.pkl', "rb") as f:
    words = pickle.Unpickler(f).load()

# p = bow("How many hosts in esxi", words)
#
# inputvar = pd.DataFrame([p], dtype=float, index=['input'])

model = keras.models.load_model('./model-data')
# result = model.predict(inputvar)

classes = None
maps = {}
with open('./classes.pkl', "rb") as f:
    classes = pickle.Unpickler(f).load()
with open('./contexts.pkl', "rb") as f:
    maps = pickle.Unpickler(f).load()


def classify_local(sentence):
    ERROR_THRESHOLD = 0.25

    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words, False)], dtype=float, index=['input'])

    # Predicts model and provided corresponding probabilities
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    return return_list



