# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 14:47:53 2015

@author: Ievgeniia
"""

import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        return ('Writer (id=%s, forms=%s, lines=%s)' 
                % (repr(self.id), repr(len(self.formsRef)), repr(self.savedSumLines) ))
    
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
# input - image of one line. Output: padded on top and bott, cutted on side
def padWithZerosAndCut(i_line, i_maxHeight, i_minWidth):
    # cut side
    line = i_line[:, 0:i_minWidth]
    curHeight = line.shape[0]
    padOnTop = (i_maxHeight - curHeight) / 2
    padOnBot = i_maxHeight - curHeight - padOnTop
    # pad on top and bottom
    line = np.pad(line, ((padOnTop, padOnBot), (0, 0)), mode='constant')[:, :]
#    print line.shape
    return line

#####################################################################
# input and output : dict (idwriter1: all his lines, idwriter2: all his lines)
def preprocessImages(linesToTrain, minAcceptableWidth):
    print '------'
    print 'Preprocessing images for', len(linesToTrain), 'writers:'
    
    lineHeightsPerWriter = map(lambda x:map(lambda y:y.shape[0], x), linesToTrain.values())
    lineWidthsPerWriter = map(lambda x:map(lambda y:y.shape[1], x), linesToTrain.values())
    # flatten array to get min and max later
    lineWidths = list(itertools.chain(*lineWidthsPerWriter))
    lineHeights = list(itertools.chain(*lineHeightsPerWriter))
    
    print 'Lines with width less than ', minAcceptableWidth, 'will be rejected'
    willBeNotRejected = [ _ for _ in itertools.compress(lineWidths, map(lambda x: x>=minAcceptableWidth, lineWidths)) ]
    
    maxHeight = max(lineHeights)
    minWidth = max(minAcceptableWidth, min(willBeNotRejected))
    print 'Width of images -', minWidth
    print 'Height of images -', maxHeight
    
    # reject lines that shorter than minAcceptableWidth and pad others
    for wId in linesToTrain.keys():
        linesForCurWriter = len(linesToTrain[wId])
        linesToTrain[wId] = [padWithZerosAndCut(l, maxHeight, minWidth) for l in linesToTrain[wId] if l.shape[1] > minAcceptableWidth]
        print 'Rejected', linesForCurWriter - len(linesToTrain[wId]), 'lines for writer ', wId, '( out of',linesForCurWriter,')'
    
    print 'Preprocessing done'
    return linesToTrain
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
#print sortedWriters[0:10]

numWritersToTrain = 5
# load forms, lines and words images for writers
# and create dict (idwriter1: all his lines, idwriter2: all his lines)
linesToTrain = {}
for item in sortedWriters[1:numWritersToTrain+1]: # first wrote too much
    writer = item[1]
    linesToTrain[writer.id] = []
    print 'Loading lines for writer ', writer.id, 'from', len(writer.formsRef), ' forms...'
    for formId in writer.formsRef:
        for lineId in forms[formId].linesRef:
            lines[lineId].loadData()
#            # for debug: take only few lines per writer
#            if (len(linesToTrain[writer.id]) >= 2):
#                continue
            linesToTrain[writer.id].append(lines[lineId].data)
################################################################################

minAcceptableWidth = 1000
linesToTrain = preprocessImages(linesToTrain, minAcceptableWidth)
#  len(linesToTrain.items()[0][1])
# plt.imshow(linesToTrain.items()[0][1][0])

        