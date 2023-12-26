# The main program, using all data_preprocessing,py, data_analysis.py 
# and data_visualization.py to train or re-train the model.

# libraries

from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# use the variables in data_preprocessing.py
from data_preprocessing import chinese_vocab, english_vocab, max_chinese_len, max_english_len, chi_pad_sentence, eng_pad_sentence, tokenize

# Encoder
input_sequence = Input(shape=(max_english_len,))
embedding = Embedding(input_dim=english_vocab, output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)

# Decoder
input_sequence = Input(shape=(max_english_len,))
embedding = Embedding(input_dim=english_vocab, output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_chinese_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(chinese_vocab))(decoder)

# Layer Stacking
enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(1e-3),
              metrics=['accuracy'])
enc_dec_model.summary()

# Model
model_results = enc_dec_model.fit(eng_pad_sentence, chi_pad_sentence, batch_size=90, epochs=2)


# # Answer
# def logits_to_sentence(logits, tokenizer):

#     index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
#     index_to_words[0] = '<empty>' 

#     return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# input_sentence = "The steps of data preprocessing should be included in this file."
# print("The english sentence is: {}".format(input_sentence))
# print('The predicted sentence is :')
# eng_text_tokenized, text_tokenizer = tokenize(input_sentence)
# print(logits_to_sentence(enc_dec_model.predict(eng_text_tokenized, text_tokenizer)))
