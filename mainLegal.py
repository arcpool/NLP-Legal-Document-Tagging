#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:20:04 2021

@author: arya
"""
import time
start = time.time()
import os
from spacy.pipeline import EntityRuler
import en_core_web_sm
from multi_rake import Rake

os.chdir('legalData/Train_tags')
tagFiles =  os.listdir()

def sortTagsByCase(x):
    return(x[-8:])

tagFiles = sorted(tagFiles, key = sortTagsByCase) #sorted tag file names of train tags by case number

os.chdir('/Users/arya/Desktop/DocumentTagging/legalData/Train_docs')
trainFiles =  os.listdir()

def sortDocsByCase(y):
    return(y[-19:])

trainFiles = sorted(trainFiles, key = sortDocsByCase) #sorted doc file names of train docs by case statement number
trainFiles.pop(0)

trainTags = [] #to store given tags in a list form from tagFiles
os.chdir('/Users/arya/Desktop/DocumentTagging/legalData/Train_tags')

#given tags stored in trainTags as a list
for i in range(0, (len(tagFiles))):
    raw = open(tagFiles[i]).read()
    trainTags.append(raw.split(","))

trainDocs = [] #to store key words from give docs and trainFiles in a list form 
os.chdir('/Users/arya/Desktop/DocumentTagging/legalData/Train_docs')

#important keywords from trainFiles stored in trainDocs as a list
rake = Rake(min_chars = 4,
    max_words = 3,
    min_freq = 2,
    language_code=None,  # 'en'
    stopwords=None,  # {'and', 'of'}
    lang_detect_threshold=50,
    max_words_unknown_lang=2,
    generated_stopwords_percentile=80,
    generated_stopwords_max_len=3,
    generated_stopwords_min_freq=2,
)

docs = [] #to store opened trainFiles

for i in range(0,len(trainFiles)):
    raw = open(trainFiles[i], encoding = "latin-1")
    txt = raw.read()
    docs.append(txt) #opening each file in trainFiles and storing it in docs
    
wordsByRake =[] #to store key words from each docs after applying Rake

for i in range(0,len(trainFiles)):
    wordsByRake.append(rake.apply(docs[i]))

def listOfLists(lst):
    temp =[]
    for i in range(0,len(lst)):
        if(lst[i][1] > 4.0): #this can be changed according to the required number of tags
            temp.append(lst[i][0])
    return [elem for elem in temp]

for i in range(0,len(wordsByRake)):
    trainDocs.insert(i, listOfLists(wordsByRake[i]))

wordsByRake = [] #to avoid memory usage
docs = [] #to avoid memory usage

nlp = en_core_web_sm.load() #loading the model
ruler = EntityRuler(nlp) 

pattern = []; #to store the pattern for tagging each doc

def labelDecider(txt,tag,count):
    patternDict = {}
    patternDict = {'label': str(tag) ,'pattern': txt[count]}  
    return patternDict

for i in range(0,len(trainDocs)): #0 to 80
    docCount = 0;
    case = trainDocs[i]
    if(len(trainDocs[i]) != 0):
        if(len(trainTags[i]) == 1): #for docs with one lable
            tagCase = trainTags[i]
            while True:  
                pattern.append(labelDecider(case,tagCase,docCount))
                docCount = docCount + 1
                if(docCount == len(trainDocs[i])):  
                    break   
        
        else: #for docs with multiple labels: matching each label to each keyword of the doc
            for j in range(0,len(trainTags[i])): 
                docCount = 0
                tags = trainTags[i]
                while True:  
                    pattern.append(labelDecider(trainDocs[i],tags[j],docCount))
                    docCount = docCount + 1
                    if(docCount == len(trainDocs[i])):  
                        break

#adding the pattern to the pipe
ruler.add_patterns(pattern)
nlp.add_pipe(ruler)

# disabling default pipe names to add custom : ['tagger', 'parser', 'ner'] disabled    
pipe_exceptions = ['entity_ruler']
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
nlp.disable_pipes(*unaffected_pipes) 

#applying the model
testPath = '/Users/arya/Desktop/DocumentTagging/testData/Test_docs/'
os.chdir(testPath)
testFiles =  os.listdir()

def sortTestByCase(x):
    return(x[-17:])

testFiles = sorted(testFiles, key = sortTestByCase) #sorted test file names of train tags by case number
testFiles.pop(0)

testDocs = []
for i in range(0,len(testFiles)):
    testDocs.append(open(testFiles[i], encoding = "latin-1").read())

separateTags = []
def tagSeparator(tup):
    separateTags.append(tup)
    
finalTags = [] #to store the final tags generated for the test files

for i in range(0,len(testDocs)):
    postModelTest = [nlp(text) for text in [testDocs[i]]]
    for j in range(0,len(postModelTest)):
        separateTags = []
        for k in range(len(postModelTest[j].ents)):
            tagSeparator((postModelTest[j].ents[k].text,postModelTest[j].ents[k].label_))
        finalTags.append(separateTags)

#exporting tags as a csv file
import pandas as pd
df = pd.DataFrame(finalTags)
os.chdir('/Users/arya/Desktop/DocumentTagging/')
df.to_csv('outputTags.csv',header=False,index=False)
        
end = time.time()
print(end - start)

#training the model
#import random
#from spacy.util import minibatch, compounding
##from pathlib import Path
#
## TRAINING THE MODEL
#with nlp.disable_pipes(*unaffected_pipes):
#
#  # Training for 30 iterations
#  for iteration in range(30):
#
#    # shuufling examples  before every iteration
#    random.shuffle(TRAIN_DATA)
#    losses = {}
#    # batch up the examples using spaCy's minibatch
#    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
#    for batch in batches:
#        texts, annotations = zip(*batch)
#        nlp.update(
#                    texts,  # batch of texts
#                    annotations,  # batch of annotations
#                    drop=0.5,  # dropout - make it harder to memorise data
#                    losses=losses,
#                )
#        print("Losses", losses)
