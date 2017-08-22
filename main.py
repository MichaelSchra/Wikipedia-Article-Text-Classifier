# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:02:51 2016
@author: Michael Schra

NOTES:
-Extendable to more than two classes by adding categories to the classes list
-Change the refresh_data flag to true to refresh the Wikipedia data
"""
#Developer Variables
classes = ['Superhero_film_characters', 'Theoretical_physicists']
refresh_data = False        #Set to True to refresh the Wikipedia data

import getWikiData as WD         #Custom module to load and parse Wikipedia data
import getClassifierModels as CM #Contains the code for document classification

#Due to the long amount of time it takes to download all the Wikipedia data
#from the API then parse the data, the default is to WD.load_data() 
#from local cache.
print("Loading and cleaning the Wikipedia Data. This may take a few minutes.")
articles = WD.get_Wiki_Data(classes) if refresh_data else WD.load_data(classes)

print("Creating the article classification models. This will take a few minutes...")
#Returns all Unique links for  feature selection
all_links = CM.getLinks(articles)   
#Returns the top 20% most common words for feature selection
common_words = CM.getCommonWords(articles)
words_data_set = CM.getDataSet_words(articles, common_words)
links_data_set = CM.getDataSet_links(articles, all_links)
Wclassifier = CM.getNaiveBayesClassifier(words_data_set)
Lclassifier = CM.getNaiveBayesClassifier(links_data_set)

while True:
    #Get A Wikipedia Article's URL from the user, and lookup the pageid
    #WARNING: This logic doesn't work for a URL that auto redirects to another page
    pageid = '-1'          #What Wikipedia's API returns if it doesn't match a URL
    request = input("\nEnter a Wikipedia article's URL for classification (quit to exit): ")
    if request.lower() in ['quit', 'exit', 'q']:  break  
    elif '/' in request:  
        pageid = WD.getPageID(request[request.rfind("/") + 1:])
    elif pageid == '-1':
        print('Invalid Wikiedpia URL. Please try again.')
        continue
    
    #Check if in existing classes
    pageids = [str(a['pageid']) for a in articles]
    if pageid in pageids:
        for a in articles:
            if pageid == str(a['pageid']):
                print("\nThis Wikipedia article is in the " + a['class'] + " category.")                
                break
    else:
        print("\nThis Wikipedia article is not in either of the categories.")
    
    #Classify by existing models
    WC_guess = Wclassifier.classify(CM.getFeatures(WD.getWords(pageid), common_words, "Contains"))
    LC_guess = Lclassifier.classify(CM.getFeatures(WD.getLinks(pageid), all_links, "LinksTo"))
    print("The Word Classifier Model predicts the " + WC_guess + " category.")
    print("The Link Classifier Model predicts the " + LC_guess + " category.")
    
    
'''
#Testing Code
import nltk                      #Choice of NLP Python packages

train_ratio = .75
Wclassifier = CM.getNaiveBayesClassifier(words_data_set[:int(len(articles)*train_ratio)])
print("Word Classifier accuracy percent:",(nltk.classify.accuracy(Wclassifier, words_data_set[int(len(articles)*train_ratio):]))*100)
Wclassifier.show_most_informative_features(15)

Lclassifier = CM.getNaiveBayesClassifier(links_data_set[:int(len(articles)*train_ratio)])
print("Link Classifier accuracy percent:",(nltk.classify.accuracy(Lclassifier, links_data_set[int(len(articles)*train_ratio):]))*100)
Lclassifier.show_most_informative_features(15)

def confusion_matrix(data):
    #Manually switched words/links to test each model
    test_errors_S = []
    test_errors_T = []
    test_correct_S =[]
    test_correct_T = []
    for a in data:
        links = a['links']
        cat = a['class']
        guess = Lclassifier.classify(CM.getFeatures(links, all_links, "LinksTo"))
        if guess == cat:
            if cat == 'Superhero_film_characters': test_correct_S.append((a))
            if cat == 'Theoretical_physicists': test_correct_T.append((a))
        else:
            if cat == 'Superhero_film_characters': test_errors_S.append((a))
            if cat == 'Theoretical_physicists': test_errors_T.append((a))
    return test_correct_S, test_correct_T, test_errors_S, test_errors_T
        
test_correct_S, test_correct_T, test_errors_S, test_errors_T = confusion_matrix(articles)

#End of Test Code 
'''

