from hazm import *
import re
import string

def remove_hyperlink(sentence):
    sentence = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", sentence[::-1])
    sentence = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", sentence[::-1])
    return sentence

def to_lower(sentence):
    return sentence.lower()

def remove_number(sentence):
    return re.sub(r'\d+', '', sentence)

def remove_punctuation(sentence):
    sentence = sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return sentence.replace("،", "")
    
def remove_whitespace(sentence):
    return sentence.strip()

def replace_newline(sentence):
    return sentence.replace('\n','')

def remove_stop_words_and_lammatize(sentence):
    lemmatizer = Lemmatizer()
    stop_words = stopwords_list()
    tokens = word_tokenize(sentence)
    new_tokens = [lemmatizer.lemmatize(x) for x in tokens if x not in stop_words]
    new_sentence = ' '.join(new_tokens)
    return new_sentence

def normalize(sentence):
    normalizer = Normalizer()
    return normalizer.normalize(sentence)

def clean_up_pipeline(text):
    cleaning_utils = [to_lower,
                      remove_hyperlink,
                      replace_newline,
                      remove_number,
                      remove_punctuation,
                      remove_whitespace,
                      normalize,
                      remove_stop_words_and_lammatize
                     ]
    for o in cleaning_utils:
        text = o(text)
    return text