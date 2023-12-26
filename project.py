import string
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import legacy
from keras.losses import sparse_categorical_crossentropy


# Path to translation file
path_to_data = 'data/cmn.txt'

# Read file
translation_file = open(path_to_data,"r", encoding='utf-8') 
raw_data = translation_file.read()
translation_file.close()

# Parse data
raw_data = raw_data.split('\n')
pairs = [sentence.split('\t') for sentence in raw_data]
pairs = pairs[0:-1]

def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()
    # Strip punctuation
    chinese_punctuation = "。" + "【" + "】"
    string_punctuation = string.punctuation + chinese_punctuation
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
   
    return clean_sentence


def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

# Clean sentences
english_sentences = [clean_sentence(pair[0]) for pair in pairs]
chinese_sentences = [clean_sentence(pair[1]) for pair in pairs]

# Tokenize words
chi_text_tokenized, chi_text_tokenizer = tokenize(chinese_sentences)
eng_text_tokenized, eng_text_tokenizer = tokenize(english_sentences)

print('Maximum length chinese sentence: {}'.format(len(max(chi_text_tokenized,key=len))))
print('Maximum length english sentence: {}'.format(len(max(eng_text_tokenized,key=len))))


# Check language length
chinese_vocab = len(chi_text_tokenizer.word_index) + 1
english_vocab = len(eng_text_tokenizer.word_index) + 1
print("Chinese vocabulary is of {} unique words".format(chinese_vocab))
print("English vocabulary is of {} unique words".format(english_vocab))

max_chinese_len = int(len(max(chi_text_tokenized,key=len)))
max_english_len = int(len(max(eng_text_tokenized,key=len)))

chi_pad_sentence = pad_sequences(chi_text_tokenized, max_chinese_len, padding = "post")
eng_pad_sentence = pad_sequences(eng_text_tokenized, max_english_len, padding = "post")

# Reshape data
chi_pad_sentence = chi_pad_sentence.reshape(*chi_pad_sentence.shape, 1)
eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

# English -> Chinese
input_sequence = Input(shape=(max_english_len,))
embedding = Embedding(input_dim=english_vocab, output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_chinese_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(chinese_vocab))(decoder)

enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy, optimizer=legacy.Adam(1e-3), metrics=['accuracy'])
enc_dec_model.summary()

results = enc_dec_model.fit(eng_pad_sentence, chi_pad_sentence, batch_size=60, epochs=250)
# Save the trained model
enc_dec_model.save("my_model.keras")


def logits_to_sentence(logits, tokenizer):

    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>' 


    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

index = 14
print("The english sentence is: {}".format(english_sentences[index]))
print("The chinese sentence is: {}".format(chinese_sentences[index]))
print('The predicted sentence is :')
print(logits_to_sentence(enc_dec_model.predict(eng_pad_sentence[index:index+1])[0], chi_text_tokenizer))
