from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Bidirectional, GRU
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import os

#setting globals
#first create a combined text and title attribute to embed
dl_set = combined.copy()
dl_set['text2'] = [(dl_set.title[x] + " " + dl_set.text[x]) for x in range(len(dl_set.text))]

labels2 = [1 if x == 'REAL' else 0 for x in labels]

maxlen = 2000
max_words = 20000
train_samples = (len(train) - 50)
validation_samples = 50
final_test_samples = 0


#tokenizing
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(dl_set['text2'])
sequences = tokenizer.texts_to_sequences(dl_set['text2'])

#creating train, val and test sets
word_index = tokenizer.word_index
print('found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen = maxlen)

x_train = data[:train_samples]
y_train = labels2[:train_samples]
x_val = data[train_samples: train_samples + validation_samples]
y_val = labels2[train_samples: train_samples + validation_samples]
x_final_test = data[train_samples + validation_samples:]
y_final_test = labels2[train_samples + validation_samples:]
