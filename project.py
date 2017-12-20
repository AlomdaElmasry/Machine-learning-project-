# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 01:09:17 2017

@author: mr root
"""
import numpy as np 
import pandas as pd
from pandas import DataFrame
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.svm import SVC ,NuSVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.pipeline import Pipeline


#load data in dataframe
def GetData():
    reader = pd.read_csv('data1.csv',encoding='utf-8')
    return reader
database=GetData()

#preprocessing
def normalizeArabic(text): #normalize el words
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text= emoji_pattern.sub(r'', text)
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                                 
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    
    return text

#remove stop words
    
def stopWordRemove(text):
    ar_stop_list=open('arabic.txt',"r",encoding='utf-8')
    stop_words=ar_stop_list.read().split('\n')
    needed_words=[]
    words=word_tokenize(text)
    for w in words:
        if w not in (stop_words):
            needed_words.append(w)
    filtered_sentence=" ".join(needed_words)
    return filtered_sentence

#steaming words

def steaming(text):
    st=ISRIStemmer()
    stemmed_words=[]
    words=word_tokenize(text)
    for w in words:
        stemmed_words.append(st.stem(w))
    stemmed_sentence=" ".join(stemmed_words)
    return stemmed_sentence

#data after preparation

def preparedata(data):
    sentences=[]
    for index ,r in data.iterrows():
        
        text=stopWordRemove(r['Tweet'])
        text=normalizeArabic(r['Tweet'])
        text=steaming(r['Tweet'])
        sentences.append([text,r["Topic"]])
    df_sentences=DataFrame(sentences,columns=["Tweet","Topic"])
    return df_sentences

preprocessed_data=preparedata(database)

#Extract Feature

def featurExtraction(data):
    tfidf_transformer=TfidfTransformer(min_df=10,max_df=.75,ngram_range=(1,3))
    tfidf_data=tfidf_transformer.fit_transform(data)
    return tfidf_data


#Learning
    
def learning(clf,x,y):
    x_train,x_test,y_train,y_test=\
    cross_validation.train_test_split(x,y,test_size=.97,random_state=500)
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', clf(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
    ])
    classifer =text_clf.fit(x_train,y_train)
    new_tweet=["adsl !!!!!!"]
    X_new_counts = count_vect.transform(new_tweet)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    New_predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, New_predicted):
        print('%r => %s' % (doc, y_train.target_names[category]))
    predict=cross_validation.cross_val_predict(classifer,x_test,y_test,cv=20)
    scores=cross_validation.cross_val_score(classifer,x_test,y_test,cv=20)
    print(scores)
    print("Accuracy of %s: %0.2f (+/- %0.2f)"%(classifer,scores.mean(),scores.std()*2))
    print(classification_report(y_test,predict))


#Main DEF 
def main(clf):
    database=GetData()
    preprocessed_data=preparedata(database)
    data,target=preprocessed_data["Tweet"],preprocessed_data["Topic"]
    tfidf_data=featurExtraction(data)
    learning(clf,tfidf_data,target)
    
        
main(SGDClassifier)



























