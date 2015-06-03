# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 14:47:53 2015

@author: Ievgeniia
"""

import os
from scipy import misc

DBpath = os.path.join(".", 'IAM')
formsPath = os.path.join(DBpath, 'forms')
linesPath = os.path.join(DBpath, 'lines')
wordsPath = os.path.join(DBpath, 'words')
asciiPath = os.path.join(DBpath, 'ascii')
formsFile = os.path.join(asciiPath, 'forms.txt')
linesFile = os.path.join(asciiPath, 'lines.txt')
wordsFile = os.path.join(asciiPath, 'words.txt')

#####################################################################
class Writer:
    
    def __init__(self, i_id = -1):
        self.id = i_id
        self.formsRef = []
        self.savedSumLines = -1;
        self.savedSumWords = -1;
        
    def __repr__(self):
        return ('Writer (id=%s, forms=%s, words=%s)' 
                % (repr(self.id), repr(len(self.formsRef)), repr(self.savedSumWords) ))
    
    def sumLines(self):
        if (self.savedSumLines != -1):
            return self.savedSumLines
        sum = 0
        for formId in self.formsRef:
            sum += forms[formId].sumLines()
        return sum
        
    def sumWords(self):
        if (self.savedSumWords != -1):
            return self.savedSumWords
        sum = 0
        for formId in self.formsRef:
            sum += forms[formId].sumWords()
        return sum
        
    def calculateAnsSaveSums(self):
        self.savedSumLines = self.sumLines()
        self.savedSumWords = self.sumWords()
                

#####################################################################
class Form:
    
    def __init__(self, i_id = ''):
        self.id = i_id
        self.linesRef = []
        self.data = []
        
    def sumLines(self):
        return len(self.linesRef)
        
    def sumWords(self):
        sum = 0
        for lineId in self.linesRef:
            sum += lines[lineId].sumWords()
        return sum
        
    def __repr__(self):
        return ('Form (id=%s, lines=%s)' 
                % (repr(self.id), repr(self.sumLines()) ) )
                
    def loadData(self, i_formsPath = formsPath):
        fullFileName = os.path.join(i_formsPath, self.id + '.png')
        self.data = misc.imread(fullFileName)
 
#####################################################################       
class Line:
    
    def __init__(self, i_id = '', i_label = ''):
        self.id = i_id
        self.wordsRef = []
        self.label = i_label
        self.data = []
        
    def sumWords(self):
        return len(self.wordsRef)
        
    def __repr__(self):
        return ('Line (id=%s, words=%s, label=%s)' 
                % (repr(self.id), repr(self.sumWords()),  repr(self.label)))
                
    def loadData(self, i_linesPath = linesPath):
        partsName = self.id.split('-');
        folder1 = os.path.join(i_linesPath, partsName[0])
        folder2 = os.path.join(folder1, partsName[0] + '-' + partsName[1])
        fullFileName = os.path.join(folder2, self.id + '.png')
        self.data = misc.imread(fullFileName)

#####################################################################        
class Word:
    def __init__(self, i_id = '', i_label = ''):
        self.id = i_id
        self.label = i_label
        self.data = []
        
    def __repr__(self):
        return ('Word (id=%s, label=%s)' 
                % (repr(self.id),  repr(self.label)))
    
    def loadData(self, i_wordsPath = wordsPath):
        partsName = self.id.split('-');
        folder1 = os.path.join(i_wordsPath, partsName[0])
        folder2 = os.path.join(folder1, partsName[0] + '-' + partsName[1])
        fullFileName = os.path.join(folder2, self.id + '.png')
        self.data = misc.imread(fullFileName)
        
#####################################################################
#####################################################################
writers = {}
forms = {}
lines = {}
words = {}


#%% read forms
f=open(formsFile)
for line in f:
    line = line.strip()
    if (line[0] is '#'):
        continue

    columns = line.split()
    formId = columns[0]
    writerId = int(columns[1])
    
    if (not writers.has_key(writerId)):
        writers[writerId] = Writer(writerId)

        
    writer = writers[writerId]
    writer.formsRef.append(formId)
    
    newForm = Form(formId)
    forms[formId] = newForm
    
f.close()

#%% read lines

f=open(linesFile)
for line in f:
    line = line.strip()
    if (line[0] is '#'):
        continue

    columns = line.split()
    lineId = columns[0]
    label = columns[8]

    lines[lineId] = Line(lineId, label)
    
    # add this line to corresponding form
    formId = lineId[0:-3]
    forms[formId].linesRef.append(lineId)
f.close()

#%% read words

f=open(wordsFile)
for line in f:
    line = line.strip()
    if (line[0] is '#'):
        continue
    
    columns = line.split()
    wordId = columns[0]
    label = columns[8]

    words[wordId] = Word(wordId, label)
    
#    print wordId
    
    # add this word to corresponding line
    lineId = wordId[0:-3]
    lines[lineId].wordsRef.append(wordId)
f.close()

#%% save how num of Lines and words
for writerId in writers:
    writers[writerId].calculateAnsSaveSums()
    
#%%
sortedWriters = sorted(writers.items(), key=lambda w: w[1].savedSumWords, reverse=True)
# only 50 writers wrote more than 400 words
print sortedWriters[0:10]

#test
#words.items()[8][1].loadData(wordsPath)

trainWriters = sortedWriters[0:5]
# load forms, lines and words images for writers
for item in trainWriters:
    writer = item[1]
    print 'Loading ', len(writer.formsRef), ' forms...'
    for formId in writer.formsRef:
        forms[formId].loadData()
        for lineId in forms[formId].linesRef:
            lines[lineId].loadData()
#            print '  Loading ', len(lines[lineId].wordsRef), ' words'
            for wordId in lines[lineId].wordsRef:
                words[wordId].loadData()
                

#%% data was read
###########################################################################################
## in same way images will be accessed for each writer and tuples will be created
#for item in trainWriters:
#    writer = item[1]
#    for formId in writer.formsRef:
#        # forms[formId] - work with forms for writer
#        for lineId in forms[formId].linesRef:
#            # lines[lineId] - work with lines for writer
#            for wordId in lines[lineId].wordsRef:
#                # words[wordId] - work with words for writer
