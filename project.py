# Student ID: 1155158477
# Name: YAU YUK TUNG


import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import legacy
from keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences
from data_preprocessing import chinese_vocab, english_vocab, max_chinese_len, max_english_len, chi_pad_sentence, eng_pad_sentence, english_sentences, chinese_sentences, chi_text_tokenizer, tokenize, clean_sentence

def logits_to_sentence(logits, tokenizer):

    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>' 
    answer = ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    return answer

# English -> Chinese
input_sequence = Input(shape=(max_english_len,))
embedding = Embedding(input_dim=english_vocab, output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_chinese_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(chinese_vocab))(decoder)

enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy, optimizer=legacy.Adam(1e-3), metrics=['accuracy', 'mse'])
enc_dec_model.summary()

results = enc_dec_model.fit(eng_pad_sentence, chi_pad_sentence, batch_size=32, epochs=225)
# Save the trained model

# enc_dec_model.save("my_model.keras")
loss_values = results.history['loss']
accuracy_values = results.history['accuracy']
mse_values = results.history['mse']

# Print the loss and metric values for each epoch
for epoch in range(len(loss_values)):
    print(f"Epoch {epoch+1}: Loss = {loss_values[epoch]}, Accuracy = {accuracy_values[epoch]}, MSE = {mse_values[epoch]}")


index = 14
print("The english sentence is: {}".format(english_sentences[index]))
print("The chinese sentence is: {}".format(chinese_sentences[index]))
print('The predicted sentence is :')
print(logits_to_sentence(enc_dec_model.predict(eng_pad_sentence[index:index+1])[0], chi_text_tokenizer))

# m = load_model('my_model.keras')

# index = 14
# print("The english sentence is: {}".format(english_sentences[index]))
# print("The chinese sentence is: {}".format(chinese_sentences[index]))
# print('The predicted sentence is :')
# print(logits_to_sentence(m.predict(eng_pad_sentence[index:index+1])[0], chi_text_tokenizer))