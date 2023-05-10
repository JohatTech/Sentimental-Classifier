import json
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000


def read(file):
    for f in open(file, 'r'):
        yield json.loads(f)
datastore = read('./Sarcasm_Headlines_Dataset.json')

sentences  = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
oov_tok = "<00V>"


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index


#TRAIN
train_sequence = tokenizer.texts_to_sequences(training_sentences)
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequence, 
maxlen = max_length, padding =padding_type, truncating = trunc_type)

#TEST
test_sequence = tokenizer.texts_to_sequences(testing_sentences)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequence, 
maxlen = max_length, padding =padding_type, truncating = trunc_type)


#NEURAL NETWORK 
import numpy as np
training_padded = np.array(train_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(test_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.compile(loss ='binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
num_epochs = 30

history = model.fit(train_padded, training_labels, epochs = num_epochs, validation_data =(test_padded, testing_labels), verbose =2)