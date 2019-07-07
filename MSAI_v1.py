
# coding: utf-8

# In[85]:


import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.data import find
from nltk.util import ngrams
import pandas as pd
import numpy as np

import spacy
en_nlp = spacy.load('C:/Users/suvajit/Anaconda3/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')
#en_nlp = spacy.load('en')
# In[103]:


class text_clean:

    def __init__(self):
        pass
    
    def punctuation_trimming(self,sent):
        y = [x for x in sent if x not in string.punctuation]
        return y

    def special_char_removal(self,tok):
        
        z = [re.sub('[^A-Za-z0-9]+', '', token) for token in tok]
        z = [x for x in z if x]
        return z

    #Remove stop words
    stop_words = set(stopwords.words('english'))

    def stopw_rem(self,tok):
        
        clean_tokens = tok[:]
        for token in tok:
            if token in self.stop_words:
                clean_tokens.remove(token)
        return clean_tokens

    #Convert to lower case
    def conv_to_lower(slef,sent):
        newtok = [item.lower() for item in sent]
        return newtok

    #POS tagger

    PICKLE = "averaged_perceptron_tagger.pickle"
    AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))
    tagger = PerceptronTagger(load=False)
    tagger.load(AP_MODEL_LOC)
    pos_tag = tagger.tag

    #Extract Noun
    def noun_iden(self,sent):
        tok = word_tokenize(sent)
        nountok = [word for (word, pos) in self.pos_tag(tok) if pos[:2] == 'NN']
        return nountok

    # Identify the POS and lemmatize according using the parameter pos in lemmatization
    ### Lemmatization


    lemmatizer = WordNetLemmatizer()

    def lemm(self,sent):
        tok = word_tokenize(sent)
        tok2 = []
        for word,tag in self.pos_tag(tok):
            if tag.startswith("NN"):
                temp =  self.lemmatizer.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                temp =  self.lemmatizer.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                temp =  self.lemmatizer.lemmatize(word, pos='a')
            else:
                temp =  word
            tok2.append(temp)
        return ' '.join(tok2)

    ### Stemming
    ps = PorterStemmer()

    def stem(self,sent):
        newtok = [self.ps.stem(w) for w in word_tokenize(sent)]
        return ' '.join(newtok)


# In[119]:


def question_mark(x):
    if "?" in x:
        return 1
    else:
        return 0

def length_of_corpus(x):
    words = word_tokenize(x)
    words = [word.lower() for word in words if word.isalpha()]
    return len(words)

##Starts with wh words
def wh_words(x):
    x_new=x.lower()
    x_new=nltk.word_tokenize(x_new)
    return any(list(map(lambda x : x.startswith("wh"), x_new)))

##Lemmatized root word

def lemmatizer_qu(x):
    lemmatizer = WordNetLemmatizer()
    lemma=[]
    temp=[sent.root for sent in en_nlp(x).sents]
    for i in temp:
        try:
            lemma.append(lemmatizer.lemmatize(str(i).lower()))
        except:
            lemma.append(np.nan)
    return lemma

def create_ngrams(x,n):
    n_grams=[]
    words=nltk.word_tokenize(x)
    words=[word.lower() for word in words if word.isalpha()]
    for i in ngrams(words,n):
        n_grams.append(i)
    return n_grams


def count_words(x):
    counting_words=['number','average','many','often','much','percent','percentage','ratio']
    count_wrd=[]
    words=nltk.word_tokenize(x)
    words=[word.lower() for word in words if word.isalpha()]
    for i in counting_words:
        if i in words:
            return 1
        else:
            return 0


def ner(x):
    tagged=[]
    document=en_nlp(x)
    for element in document.ents:
        tagged.append((element.label_))
    return tagged

def process_question(question, en_nlp):
    en_doc = en_nlp(u'' + question)
    sent_list = list(en_doc.sents)
    sent = sent_list[0]
    wh_bi_gram = []
    root_token = ""
    wh_pos = ""
    wh_nbor_pos = ""
    wh_word = ""
    for token in sent:
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            wh_pos = token.tag_
            wh_word = token.text
            wh_nbor_pos = en_doc[min(token.i + 1,len(sent)-1)].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_
    return wh_pos, wh_word, wh_nbor_pos, root_token

def token_match(q,a):
    count = len(set(q).intersection(set(a)))
    return count