from data_preprocessing import eng_pad_sentence, chi_pad_sentence, chi_text_tokenizer, max_english_len, tokenize, clean_sentence, english_sentences, chinese_sentences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

#Load the saved model
model = load_model('my_model.keras')

# def logits_to_sentence(logits, tokenizer):

#     index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
#     index_to_words[0] = '<empty>' 

#     return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# index = 43
# print("The english sentence is: {}".format(english_sentences[index]))
# print("The chinese sentence is: {}".format(chinese_sentences[index]))
# print('The predicted sentence is :')
# print(logits_to_sentence(model.predict(eng_pad_sentence[index:index+1])[0], chi_text_tokenizer))

def logits_to_sentence(logits, tokenizer):

    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>' 


    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# Test the model
sentence = "got it"
print("The english sentence is: " + sentence)
print('The predicted chinese sentence is: ')

# print(logits_to_sentence(enc_dec_model.predict(eng_text_tokenized, text_tokenizer)))
eng_text_tokenized, text_tokenizer = tokenize(sentence)
eng_pad_sentence = pad_sequences(eng_text_tokenized, max_english_len, padding="post")
# eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

print(logits_to_sentence(model.predict(eng_pad_sentence[0:1])[0], chi_text_tokenizer))
