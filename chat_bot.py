import numpy as np
import tflearn
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import tensorflow as tf
import random
import nltk
# nltk.download('punkt')
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)
print(data)
try:
    # if any changes made to the pickle file make sure this block doesnt run
    with open('data.pickle', 'rb') as f:

        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    doc_x = []
    doc_y = []
    for value in data['intents']:
        for pattern in value['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            doc_x.append(wrds)
            doc_y.append(value['tag'])
        if value['tag'] not in labels:
            labels.append(value['tag'])
    words = [stemmer.stem(x.lower()) for x in words if x not in '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, y in enumerate(doc_x):
        bag = []
        wrds = [stemmer.stem(w) for w in y]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = output_empty[:]
        output_row[labels.index(doc_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)
training = np.array(training)
output = np.array(output)

# print(training,output)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    print('Please say something to the bot')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            print('BOT: I hope to see you again')
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print("BOT: " + random.choice(responses))
        else:
            print('BOT: I Don\'t quite get that ')

chat()
