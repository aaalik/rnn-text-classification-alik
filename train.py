'''
Source Code:
https://sourcedexter.com/tensorflow-text-classification-python/
'''
import nltk
# nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
import os, os.path

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# path joining version for other paths
DIR = 'training-surat'
LABELS = os.listdir(DIR)
NUM_LABELS = len(LABELS)
CONTENT = []
# print(LABELS,'a')
data_raw = {}
for i in range(NUM_LABELS):
    DATA_PATH = os.listdir(os.path.join(DIR, LABELS[i]))
    for x in range(len(DATA_PATH)):
        # print(DATA_PATH[x])
        f = open(os.path.join(DIR, LABELS[i], DATA_PATH[x]), "r")
        DATA_PATH[x] = f.read()
    data_raw[LABELS[i]] = DATA_PATH

    # print(f)
    json_data = json.dumps(data_raw)
    # f.close()

# initialize the stemmer
stemmer = LancasterStemmer()
# variable to hold the Json data read from the file

# read the json file and load the training data
data = data_raw
# with open('data.json') as json_data:
#     data = json.load(json_data)
#     print(data)

# get a list of all categories to train for
categories = LABELS
# print(categories)
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        # print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        # print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

# print(words)
# print(docs)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print(len(train_x[0]))

# reset underlying graph data
tf.reset_default_graph()

# Build deep neural network
# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tflearn.regression(net)

# Build recurrent neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
# model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
# model.save('model.tflearn')

# Load saved model
model.load('model.tflearn')
model.save('model.tflearn')

# let's test the mdodel for a few sentences:
# the first two sentences are used for training, and the last two sentences are not present in the training data.
sent_1 = "what time is it?"  # time
sent_2 = "Hello there!"  # greeting
sent_3 = "can you give me a sandwich?"  # sandwich
sent_4 = "you must be a couple of years older then her!"  # age

# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return (np.array(bow))

# we can start to predict the results for each of the 4 sentences
print('(', sent_1, ') is:', categories[np.argmax(model.predict([get_tf_record(sent_1)]))],'(',np.amax(model.predict([get_tf_record(sent_1)])),')')
print('(', sent_2, ') is:', categories[np.argmax(model.predict([get_tf_record(sent_2)]))],'(',np.amax(model.predict([get_tf_record(sent_2)])),')')
print('(', sent_3, ') is:', categories[np.argmax(model.predict([get_tf_record(sent_3)]))],'(',np.amax(model.predict([get_tf_record(sent_3)])),')')
print('(', sent_4, ') is:', categories[np.argmax(model.predict([get_tf_record(sent_4)]))],'(',np.amax(model.predict([get_tf_record(sent_4)])),')')
