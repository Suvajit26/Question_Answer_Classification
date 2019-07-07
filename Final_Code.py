### Load necessary packages

import math
import pickle
import os
import warnings
import pickle
import numpy as np
import pandas as pd
import nltk
import spacy
import datetime as dt
from nltk.util import ngrams
from scipy.spatial.distance import cosine
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import process
import time
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import MSAI_v1 as msai
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.data import find
from nltk.corpus import wordnet as wn

warnings.filterwarnings('ignore')
en_nlp = spacy.load('en')

# Create Functions

docIDFDict = {}
avgDocLength = 0

def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f:
        passage = line.strip().lower().split("\t")[2]
        fw.write(passage+"\n")
    f.close()
    fw.close()

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        doc = line.strip().split(delimiter)
        totalDocLength += len(doc)

        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] +0.5) / (docFrequencyDict[word] + 0.5), base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments

def GetBM25Score(x ,k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength

    query_words= x['query'].strip().lower().split(delimiter)
    passage_words = x['passage'].strip().lower().split(delimiter)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

# Create a class for feature engineering  
  
class features_1():
 
    ##Ends with Question Mark
    def question_mark(self,x):
        if "?" in x:
            return 1
        else:
            return 0
    
    ##Length of question
    def length_of_corpus(self,x):
        words = nltk.word_tokenize(x)
        words = [word.lower() for word in words if word.isalpha()]
        words=  [w for w in words if w not in stop_words]
        return len(words)
    
    ##Create N Grams
    def create_ngrams(self,x,n):
        n_grams=[]
        words=nltk.word_tokenize(x)
        words=[word.lower() for word in words if word.isalpha()]
        words=[w  for w in words if w not in stop_words]
        for i in ngrams(words,n):
            n_grams.append(i)
        return list(set(n_grams))
        
    def ngrams_match(self,x,n):
        if n==1:
            count=0
            que_ngram=list(map(str,(x['ques_unigram'])))
            sent_ngram=list(map(str,(x['sent_unigram'])))
            for j in que_ngram:
                if j in sent_ngram:
                    count=count+1
            return count
        elif n==2:
            count=0
            que_ngram=list(map(str,(x['ques_bigram'])))
            sent_ngram=list(map(str,(x['sent_bigram'])))
            for j in que_ngram:
                if j in sent_ngram:
                    count=count+1
            return count
        else:
            count=0
            que_ngram=list(map(str,(x['ques_trigram'])))
            sent_ngram=list(map(str,(x['sent_trigram'])))
            for j in que_ngram:
                if j in sent_ngram:
                    count=count+1
            return count

    def count_words(self,x):
        counting_words=['number','average','many','often','much','percent','percentage','ratio','distance',
                       'far','close','long','length']
        count_wrd=[]
        words=nltk.word_tokenize(x)
        words=[word.lower() for word in words if word.isalpha()]
        for i in counting_words:
            if i in words:
                return 1            
        return 0

# Create embeddings to be used for calculating similarity         
glove_embeddings={}
emb_dim=100
def load_embeddings():
    temp={}
    global glove_embeddings,emb_dim
    file=open("glove.6B.100d.txt","r", encoding='utf-8',errors='ignore')
    for line in file:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        temp[word]=vec
    #Add Zerovec, this will be useful to pad zeros.
        temp["zerovec"] = "0.0 "*emb_dim
    file.close()
    for key in temp.keys():
        array=temp[key].split()
        array=list(map(float, array))
        glove_embeddings[key]=array

load_embeddings()

def remove_stopwords(x):
    words = nltk.word_tokenize(x)
    words = [word.lower() for word in words if word.isalpha()]
    words=  [w for w in words if w not in stop_words]
    return words

def get_embeddings(x):
    embeddings=[]
    for i in x:
        if i in glove_embeddings:
            embeddings.append(glove_embeddings[i])
    return embeddings


GetCorpus('data.tsv','corpus.tsv')
IDF_Generator('corpus.tsv', delimiter=' ', base=math.e)

# Create more functions on features

## Returns the POS of the wh question, neighbour pos of the question and the type of question
def process_question(en_doc):
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
    return wh_pos, wh_word, wh_nbor_pos

## Returns the root word
def root_word(en_doc):
    sent_list = list(en_doc.sents)
    sent = sent_list[0]
    for token in sent:
        if token.dep_ == "ROOT":
            root_token = token.tag_
            root = token.text
    return root,root_token

## Named entities  
def ner(en_doc):
    ner_word = []
    ner_label = []
    document=en_doc
    for element in document.ents:
        ner_word.append((element))
        ner_label.append((element.label_))
    return list(set(ner_label))

col=features_1()

## Calculate the cosing similarity
def cos_sim(a,b):
    try:
        ans = cosine(a,b)
    except:
        ans = 0
    return ans

def new_features(data):
    
    data.loc[:,'qtok'] = data.loc[:,'query'].apply(lambda x : word_tokenize(x))
    data.loc[:,'atok'] = data.loc[:,'passage'].apply(lambda x : word_tokenize(x))

    data.loc[:,'qtok'] = data.loc[:,'qtok'].apply(lambda x : clean_sent.special_char_removal(x))
    data.loc[:,'atok'] = data.loc[:,'atok'].apply(lambda x : clean_sent.special_char_removal(x))

    data.loc[:,'qtok'] = data.loc[:,'qtok'].apply(lambda x : clean_sent.punctuation_trimming(x))
    data.loc[:,'atok'] = data.loc[:,'atok'].apply(lambda x : clean_sent.punctuation_trimming(x))

    data.loc[:,'qtok'] = data.loc[:,'qtok'].apply(lambda x : clean_sent.conv_to_lower(x))
    data.loc[:,'atok'] = data.loc[:,'atok'].apply(lambda x : clean_sent.conv_to_lower(x))

    data.loc[:, 'query'] = data.loc[:,'qtok'].apply(lambda x : ' '.join(x))
    data.loc[:, 'passage'] = data.loc[:,'atok'].apply(lambda x : ' '.join(x))

    data['ques_new']=data['query'].apply(lambda x : remove_stopwords(x))
    data['sent_new']=data['passage'].apply(lambda x: remove_stopwords(x))
    data['ques_embeddings']=data['ques_new'].apply(lambda x: get_embeddings(x))
    data['sent_embeddings']=data['sent_new'].apply(lambda x: get_embeddings(x))
    data['ques_embeddings']=data['ques_embeddings'].apply(lambda x: np.array(x).mean(axis=0))
    data['sent_embeddings']=data['sent_embeddings'].apply(lambda x: np.array(x).mean(axis=0))
    data['cos_similarity']=data.apply(lambda x: cos_sim(x['ques_embeddings'],x['sent_embeddings']), axis=1)
    data['eu_similarity']=data.apply(lambda x: np.sqrt(sum((x['ques_embeddings']-x['sent_embeddings'])**2)), axis=1)

    del data['ques_embeddings']
    del data['sent_embeddings']
    del data['ques_new']
    del data['sent_new']

    ##Question features

    data['ques_mark']=data['query'].apply(lambda x: col.question_mark(x))
    data['ques_length']=data['query'].apply(lambda x: col.length_of_corpus(x))
    data['ques_unigram']=data['query'].apply(lambda x: col.create_ngrams(x,1))
    data['ques_bigram']=data['query'].apply(lambda x: col.create_ngrams(x,2))
    data['ques_trigram']=data['query'].apply(lambda x: col.create_ngrams(x,3))
    data['ques_cnt_wrds']=data['query'].apply(lambda x: col.count_words(x))

    ##Sentence features
    data['sent_length']=data['passage'].apply(lambda x: col.length_of_corpus(x))
    data['sent_unigram']=data['passage'].apply(lambda x: col.create_ngrams(x,1))
    data['sent_bigram']=data['passage'].apply(lambda x: col.create_ngrams(x,2))
    data['sent_trigram']=data['passage'].apply(lambda x: col.create_ngrams(x,3))

    ##Similarity features
    data['unigrams_match']=data.apply(lambda x: col.ngrams_match(x,1), axis=1)
    data['bigrams_match']=data.apply(lambda x: col.ngrams_match(x,2), axis=1)
    data['trigrams_match']=data.apply(lambda x: col.ngrams_match(x,3), axis=1)
    data['ratio_length']=data['sent_length']/data['ques_length']
    data['bm25']=data.apply(lambda x: GetBM25Score(x), axis=1)

    del data['ques_unigram']
    del data['ques_bigram']
    del data['ques_trigram']
    del data['sent_unigram']
    del data['sent_bigram']
    del data['sent_trigram']


    data['doc'] = data['query'].apply(lambda x : en_nlp(x))
    data['doc2'] = data['passage'].apply(lambda x : en_nlp(x))
    data['pq'] = data['doc'].apply(lambda x : process_question(x))
    data['ner'] = data['doc2'].apply(lambda x : ner(x))
    data['qroot'] = data['doc'].apply(lambda x : root_word(x))
    data['aroot'] = data['doc2'].apply(lambda x : root_word(x))
    data.loc[:, 'q_pos'] = data['pq'].apply(lambda x: x[0])
    data.loc[:, 'q_word'] = data['pq'].apply(lambda x: x[1])
    data.loc[:, 'qn_pos'] = data['pq'].apply(lambda x: x[2])

    data.loc[:, 'q_root'] = data['qroot'].apply(lambda x: x[0])
    data.loc[:, 'a_root'] = data['aroot'].apply(lambda x: x[0])
    data.loc[:, 'q_root_pos'] = data['qroot'].apply(lambda x: x[1])
    data.loc[:, 'a_root_pos'] = data['aroot'].apply(lambda x: x[1])
    data.loc[:, 'person'] = [1 if 'PERSON' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:, 'org'] = [1 if 'ORG' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:, 'cardinal'] = [1 if 'CARDINAL' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:, 'location'] = [1 if 'LOCATION' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:, 'gpe'] = [1 if 'GPE' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:, 'date_time'] = [1 if 'DATE' in x or 'TIME' in x else 0 for x in data.loc[:,'ner']]
    data.loc[:,'root_pos_match'] = [1 if x[0] == x[1] else 0 for x in data[['q_root_pos','a_root_pos']].values]
    data.loc[:,'root_match'] = [1 if x[0] == x[1] else 0 for x in data[['q_root','a_root']].values]
    data.loc[:, 'what'] = [1 if x == 'what' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'which'] = [1 if x == 'which' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'when'] = [1 if x == 'when' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'how'] = [1 if x == 'how' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'where'] = [1 if x == 'where' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'why'] = [1 if x == 'why' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'whose'] = [1 if x == 'whose' else 0 for x in data.loc[:,'q_word']]
    data.loc[:, 'whom'] = [1 if x == 'whom' else 0 for x in data.loc[:,'q_word']]

    return data

data = pd.read_csv('sample_data.tsv',sep='\t',header=0).iloc[:50,1:]

#data.columns = ['q_id','query','passage','label','passage_id','id']

#data = pd.read_csv('eval1_unlabelled.tsv',sep='\t',header=None).iloc[:50,:]
#data.columns = ['q_id','query','passage','passage_id']

clean_sent = msai.text_clean()


Final_data = pd.DataFrame()
start = time.time()
import itertools
if __name__ == '__main__':
    data_split = np.array_split(data, multiprocessing.cpu_count()-1)
    pool = ThreadPool(multiprocessing.cpu_count()-1)
    Final_data = pool.map(new_features, data_split)
    pool.close()
    pool.join()
    Final_data = pd.concat(Final_data)

end = time.time()
print((end - start)/60)

Final_data.to_csv('phase2_ad.tsv',sep='\t')

