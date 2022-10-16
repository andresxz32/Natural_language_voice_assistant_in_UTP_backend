from flask import Flask, request
from flask_cors import CORS

from pymongo import MongoClient
from uuid import uuid4

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import random

nltk.download('punkt')


app = Flask(__name__)
CORS(app)

# use the 'standard' representation for cross-language compatibility.
client = MongoClient(
    "mongodb+srv://andresxz32:oDQDa67GxmnBDJdj1dv13pUz7L3oyPSBi@cluster0.38tw3.mongodb.net/neuralNetwork",
    uuidRepresentation="standard",
)
collection = client.get_database("neuralNetwork").get_collection("intents")


words = []
classes = []
documents = []
ignore_words = ["?"]
# loop through each sentence in our intents patterns
intents = list(collection.find())
print(intents)
for intent in intents:
    for pattern in intent["patterns"]:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent["_id"]))
        # add to our classes list
        if intent["_id"] not in classes:
            classes.append(intent["_id"])
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique stemmed words", words)

##2

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
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

# 3

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# 4

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# 5
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


@app.route("/create_intent", methods=["POST"])
def create_intent():
    patterns = request.json["patterns"]
    responses = request.json["responses"]
    context = request.json["context"]
    uuid_obj = uuid4()
    print(uuid_obj)
    id = collection.insert_one(
        {
            "_id": str(uuid_obj),
            "patterns": patterns,
            "responses": responses,
            "context": context,
        }
    )
    return {"message": "OK"}


@app.route("/call_neural_network", methods=["POST"])
def call_neural_network():
    sentence = request.json["data"]
    acurracy = classify_local(sentence)
    idIntent = acurracy[0][0]
    acurracyScore = acurracy[0][1]
    print("Acurracy Tag:", idIntent)
    print("Acurracy Score:", acurracyScore)

    result = [x for x in intents if x["_id"] == idIntent]
    indexRes = random.randint(0, len(result[0]["responses"]) - 1)
    context = result[0]["context"]
    print(context)
    print(indexRes)
    print(result[0]["responses"][indexRes])

    if(float(acurracyScore) > 0.9):
        return {
        "Tag":idIntent,
        "Acurracy Score":acurracyScore,
        "Context":context,
        "Message":result[0]["responses"][indexRes]
        }
    else:
        return {
        "Tag":idIntent,
        "Acurracy Score":acurracyScore,
        "Context":context,
        "Message":"No se pudo encontrar una respuesta"
        }


def classify_local(sentence):
    ERROR_THRESHOLD = 0.25

    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=["input"])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability

    return return_list


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
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)



if __name__ == "__main__":
    app.run(debug=True)
