import pandas as pd
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm 
import math
from nltk.corpus import stopwords
from collections import Counter


# ---------------------------------------------------------------------------- #
#                   All the tools for preprocess the database                  #
# ---------------------------------------------------------------------------- #
STOP = stopwords.words('english')

def read_dataset(fname):
    '''
        Given a name of the file reads the dataset removing na and duplicates in the text
    '''
    df = pd.read_csv(fname, header='infer')
    df = df.iloc[df.text.drop_duplicates().index].dropna().reset_index(drop=True)
    return df

def remove_stopwords(df):
    '''
        Given a dataframe removes all the stopwords from its 'text field'
    '''
    df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in STOP)) # removing stopwords


def lower_case(df):
    '''
        Given a dataframe, makes all the words in the field 'text' lowercase
    '''
    df.text = df.text.apply(str.lower)
    df.reset_index(inplace=True, drop=True)
    df = df.iloc[df.text.drop_duplicates().index].dropna().reset_index(drop=True)


nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def lemmatize(df):
    # function to lemmatize text
    def lemmatization(text):
        lemmas = [token.lemma_ for token in nlp(text)]
        return ' '.join(lemmas)

    df['text'] = df['text'].apply(lambda x: lemmatization(x))


def get_stopword_labels(df):
    return [w.lower() in STOP for w in df.index]


def clean_document(document, good_words):
    '''
        Given a document and a white list it returns the document containing only the passed words
    '''
    return ' '.join(set(document.split(' ')).intersection(good_words))


def clean_dataset(df, good_words):
    '''
        Given a dataframe and a white list it cleans all the rows of the field 'text'
        of this datafame
    '''
    df.text = df.text.apply(lambda d: clean_document(d, set(good_words.index.values)))


def remove_empty(df): # useful function to remove empty rows
    return df[df["text"].str.len() != 0].reset_index(drop=True)


def remove_lessfreq(df):
    '''
        Given a dataset removes all the words from the field text
        that occour in total less than 5 times
    '''
    word_freq = pd.Series(' '.join(df['text']).split()).value_counts()
    word_freq = word_freq[word_freq > 4]

    def clean_document(document, words):
        return ' '.join([x for x in document.split() if x in words])

    df.text = df.text.apply(lambda d: clean_document(d, set(word_freq.index.values)))
    
    return remove_empty(df)


# --------------------------- Computing the tf-idf --------------------------- #


def compute_idf(documents):
    '''
        Given a list of documents computes the idf of each word in the documents
        and returns a dataframe having as index the words and the following fields:
            -IDF: the idf of the word
            -embedded_position: the position in the embedded vector of the word
            -is_stopword: if a word is a stopword or not
    '''
    counter = dict()
    # Counting the number of document on which every word appears
    for document in tqdm(documents):
        for word in set(document.split(' ')):
            if word !='':
                counter[word] = counter.get(word, 0)+1
    
    # computing the idf for each word
    for k,v in counter.items():
        counter[k] = math.log(len(documents)/v)
    
    # Storing the result in a dataframe with a bit of more info
    ret = pd.DataFrame.from_dict(counter, orient='index', columns=['IDF'])
    ret['embedded_position'] = np.arange(len(ret))
    ret['is_stopword'] = get_stopword_labels(ret)
    return ret


def tf_idf(document, idf_dataframe):
    '''
        Given a document and an idf dataframe having the words as index and at least two fields:
            -IDF: the idf value of the word
            -embedded_position: the position of the word in the embedded vector
        retuns an embedded vector having at each position the tf-idf value of that word
    '''

    splitted = document.split(' ')
    if '' in splitted:
        splitted.remove('')

    counted_words = Counter(splitted)
    ret = np.zeros(len(idf_dataframe))
    for word, freq in counted_words.items():
        if word == '':
            continue
        idf, idx = idf_dataframe.loc[word, ['IDF','embedded_position']]

        ret[int(idx)]=(freq/len(splitted))*idf
    return ret
