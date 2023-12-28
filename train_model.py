# Student ID: 1155158477
# Name: YAU YUK TUNG

# The main program, using all data_preprocessing,py, data_analysis.py 
# and data_visualization.py to train or re-train the model.
# libraries
import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import legacy
from keras.losses import sparse_categorical_crossentropy

# use the variables in data_preprocessing.py
from data_preprocessing import chinese_vocab, english_vocab, max_chinese_len, max_english_len, chi_pad_sentence, eng_pad_sentence, tokenize

# English -> Chinese
input_sequence = Input(shape=(max_english_len,))
embedding = Embedding(input_dim=english_vocab, output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_chinese_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(chinese_vocab))(decoder)

enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy, optimizer=legacy.Adam(0.005), metrics=['accuracy', 'mse'])
enc_dec_model.summary()

results = enc_dec_model.fit(eng_pad_sentence, chi_pad_sentence, batch_size=64, epochs=100)

enc_dec_model.save("my_model.keras")

print("Training completed.")