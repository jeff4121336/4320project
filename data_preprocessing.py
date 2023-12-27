# Student ID: 1155158477
# Name: YAU YUK TUNG

# The steps of data preprocessing should be included in this file.
# Reference: https://towardsdatascience.com/how-to-build-an-encoder-decoder-translation-model-using-lstm-with-python-and-keras-a31e9d864b9b
import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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