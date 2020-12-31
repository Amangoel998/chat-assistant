import nltk
import random
import numpy as np
import pickle

# Need to run this command whenever this scripts in ported into new system
# nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

words = []
classes = []
documents = []
# Load Stemmed words into pickle file
with open('./words.pkl', "rb") as f:
    words = pickle.Unpickler(f).load()
with open('./classes.pkl', "rb") as f:
    classes = pickle.Unpickler(f).load()
with open('./documents.pkl', "rb") as f:
    documents = pickle.Unpickler(f).load()

training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Save trained datasets into pickle file
with open('./train_x.pkl', "wb") as f:
    pickle.dump(train_x, f)
with open('./train_y.pkl', "wb") as f:
    pickle.dump(train_y, f)
