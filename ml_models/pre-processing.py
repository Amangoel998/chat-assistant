# things we need for NLP

import json
import nltk
import pickle

# Need to run this command whenever this scripts in ported into new system
# nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

words = []
classes = []
documents = []
maps = {}
responses = {}
# ignore_words = ['?', '!', '.', '-']

json_data = open('./statement_corpus.json', 'r')
intents = json.load(json_data)
# def create():
    # loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern, language="english")
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    # print(words)

    # stem and lower each word and remove duplicates
    stemmer.stem(words[0], )
    words = [stemmer.stem(w.lower(), ) for w in words if w.isalnum()]
    words = sorted(list(set(words)))

def get_more_info():
    json_data = open('./statement_corpus.json', 'r')
    intents = json.load(json_data)
    for intent in intents['intents']:
        if(len(intent['contexts'])>0):
            for cntx in intent['contexts']:
                maps.update({cntx: intent['responses'][0]})
        if(intent['tag']=='noanswer'):
            maps.update({'noanswer': intent['responses']})
        responses.update({intent['tag']: intent['responses']})

get_more_info()
# create()
# Save Stemmed words into pickle file
with open('./words.pkl', "wb") as f:
    pickle.dump(words, f)
with open('./classes.pkl', "wb") as f:
    pickle.dump(classes, f)
with open('./documents.pkl', "wb") as f:
    pickle.dump(documents, f)
with open('./contexts.pkl', "wb") as f:
    pickle.dump(maps, f)
with open('./responses.pkl', "wb") as f:
    pickle.dump(responses, f)
