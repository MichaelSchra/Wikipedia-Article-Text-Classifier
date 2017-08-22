# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:00:36 2016
@author: Michael Schra

NOTES:
-This module only contains functions called by main.py
"""
import nltk                 #Choice of NLP Python packages
import random

def getLinks(articles):
    #Returns lists of all links from the articles
    all_links = []
    for a in articles:
        all_links += a['links']
    all_links = set(all_links)
    return all_links

def getCommonWords(articles):
    #Returns list of all common words from the articles
    all_words = []
    for a in articles:
        all_words += a['words']
    
    all_words = nltk.FreqDist(all_words)
    #Extracts the 20% most common words - Enough for a good sample size (~ >10 occurances)
    common_words=[w[0] for w in all_words.most_common(int(0.2*len(set(all_words))))] 
    
    return common_words
    
def getFeatures(article_items, all_items, feature):
    features = {}
    article_items = set(article_items)  #Sets are faster to search through  
    for item in all_items:
        #Adds a lot of overhead, and very little accuracy
        #features["count({})".format(word)] = words.count(word)
        features[str(feature+"({})").format(item)] = (item in article_items)
    return features

def getDataSet_words(articles, common_words):
    data_set = [(getFeatures(a['words'], common_words, 'Contains'), a['class']) for a in articles]
    random.shuffle(data_set)
    return data_set
    
def getDataSet_links(articles, all_links): 
    data_set = [(getFeatures(a['links'], all_links, 'LinksTo'), a['class']) for a in articles]
    random.shuffle(data_set)
    return data_set

def getNaiveBayesClassifier(data):
    return nltk.NaiveBayesClassifier.train(data)



