# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:05:23 2016
@author: Michael Schra

NOTES:
-Contains functions for getting, parsing, and saving Wikipedia data
-There is an easy Wikipedia API wrapper Python package, simply called 
 'wikipedia' however I choose not to use it to demonstrate my ability 
 to use a website's API, and parse the text.
"""
import requests                        #Used for its API functions
import pickle                          #Used to save the Wikipedia data locally
import re, nltk                        #Used for text manipulation
from nltk.corpus import stopwords      #Used to remove common words
try:
    stopwords.words('english')
except:
    nltk.download("stopwords")
wiki_API_URL = 'https://en.wikipedia.org/w/api.php'

#Download and parse the Wikipedia data
def get_Wiki_Data(classes):
    articles = []
    for c in classes:                       
        articles += getArticles(c)
    save_data(articles)
    return articles

#Load the Wikipedia data from locally
def load_data(classes):
    try: 
        with open('wikiData.pickle', 'rb') as f:
            return pickle.load(f)
    except:
        print("The Wikipedia data can't be loaded from the local cache. It is being downloaded again, please be patience.\n")
        return get_Wiki_Data(classes)
        
#Save the Wikipedia data locally
def save_data(articles):
    try:
        with open('wikiData.pickle', 'wb') as f:  
            pickle.dump(articles, f)
    except:
        print('There was an error saving the Wikipedia data.\n')

#Returns information for all articles in a category
def getArticles(category):
    try:
        wiki_API_params = {
            'format': 'json',           #Easily converts to Python dictionary
            'action': 'query',          #Read (versus write)
            'list': 'categorymembers',  #What is being requested
            'cmtype': 'page',           #ignores subcategories
            'cmlimit': 'max',           #The API only allows 500 results max
            'cmnamespace': '0',         #Only root pages (articles)
            'cmtitle': 'Category:'+category, #Desired Wikipedia Category
            'cmcontinue': None          #Key for 'next page' if >500 results
        }
            
        responses = []
        while True:
            response = requests.get(url=wiki_API_URL, params=wiki_API_params).json()
            responses.append(response)
            if 'continue' in response:  #Means there are more than 500 results
                wiki_API_params['cmcontinue'] = response['continue']['cmcontinue']  
            else:
                break
                                                                            
        #Combine all the API responses for each category
        i = 0                                                                    
        articles = []
        for r in responses:             #One for every 500 items (API call)
            for a in r['query']['categorymembers']: #One for every article
                article = {}
                article['pageid'] = a['pageid']
                article['title'] = a['title']
                article['class'] = category
                article['links'] = getLinks(a['pageid'])
                article['words'] = getWords(a['pageid'])
                articles.append(article)
                i += 1
                if i % 50 == 0: print (str(i) + ' ' + category + ' articles are loaded.')
        print ('All ' + category + ' articles downloaded!\n')
        return articles
        
    except:
        print ("There was an error connecting to Wikipedia's API.\n")
        return []

#Returns all links from a Wikipedia Page
def getLinks(pageid):
    try:
        wiki_API_params = {
            'format': 'json',           #Easily converts to Python dictionary
            'action': 'query',          #Read (versus write)
            'prop': 'links',            #What is being requested
            'pllimit': 'max',           #The API only allows 500 results max
            'plnamespace': '0',         #Only root pages (articles)
            'pageids': pageid,          #Desired Wikipedia article
            'plcontinue': None          #Key for 'next page' if >500 results
        }
            
        responses = []
        while True:
            response = requests.get(url=wiki_API_URL, params=wiki_API_params).json()
            responses.append(response)
            if 'continue' in response:  #Means there are more than 500 results
                wiki_API_params['plcontinue'] = response['continue']['plcontinue']  
            else:
                break
                                                                            
        #Combine all the API responses for each article
        links = []
        for r in responses:             #One for every 500 items (API call)
            for p in r['query']['pages'][str(pageid)]['links']: #One for every link
                links.append(p['title'])
        return links
        
    except:  
        print ("There was an error connecting to Wikipedia's API.\n")
        return []

#Formats, parses, and returns all words from a Wikipedia article
def getWords(pageid):
    try:     
        wiki_API_params = {
            'format': 'json',           #Easily converts to Python dictionary
            'action': 'query',          #Read (versus write)
            'prop': 'extracts',         #What is being requested
            'explaintext': 'true',      #Strips HTML 
            'exsectionformat': 'plain', #Strips most formatting tags 
            'pageids': pageid           #Desired Wikipedia article
        }
            
        response = requests.get(url=wiki_API_URL, params=wiki_API_params).json()
        words = response['query']['pages'][str(pageid)]['extract']
        words = re.sub('[\\n]', ' ', words)        #Replace newline characters with a space
        words = re.sub('[^A-Za-z| ]', '', words)   #Remove everything not a character or space
        words = re.sub(' +', ' ', words)           #Remove back to back spaces
        words = words.lower()                      #Convert to lower case   
        words = re.split(' ', words)               #Split string to list
        stop_words = set(stopwords.words('english')) #Remove common English words
        words = [word for word in words if word not in stop_words]     
        
        #Originally I stemmed the words, but it made the model worse
        #My assumption is that nouns are very important, and stemming can mess with nouns
        #porter = nltk.PorterStemmer()              
        #words = [porter.stem(word) for word in words]
        return words

    except:  
        print ("There was an error connecting to Wikipedia's API.\n")
        return ''

#Return the pageid for a Wikipedia article
def getPageID(title):
    try:     
        wiki_API_params = {
            'format': 'json',           #Easily converts to Python dictionary
            'action': 'query',          #Read (versus write)
            'titles': title             #Desired Wikipedia article
        }

        response = requests.get(url=wiki_API_URL, params=wiki_API_params).json()
        #Return the first pageid (only 1 title is given, so only 1 element)
        return list(response['query']['pages'])[0]
        
    except:  
        print ("There was an error connecting to Wikipedia's API or the URL was invalid.\n")
        return ''



