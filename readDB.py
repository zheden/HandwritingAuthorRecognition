
import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import caffe
import lmdb
import binascii

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
def padWithOnesAndCut(i_line, i_maxHeight, i_minWidth):
    # cut side
    line = i_line[:, 0:i_minWidth]
    curHeight = line.shape[0]
    if (curHeight < i_maxHeight):
        padOnTop = (i_maxHeight - curHeight) / 2
        padOnBot = i_maxHeight - curHeight - padOnTop
        # pad on top and bottom
        line = np.pad(line, ((padOnTop, padOnBot), (0, 0)), mode='constant', constant_values=(255))[:, :]
    else:
        line = i_line[0:i_maxHeight, 0:i_minWidth]
#    print line.shape
    return line

#####################################################################
# input and output : dict (idwriter1: all his lines, idwriter2: all his lines)
def preprocessImages(linesToTrain, i_minAcceptableWidth, i_specificWidth = -1, i_specificHeight = -1):
    print '------'
    print 'Preprocessing images for', len(linesToTrain), 'writers:'
    
    if (i_specificWidth == -1):
        lineWidthsPerWriter = map(lambda x:map(lambda y:y.shape[1], x), linesToTrain.values())
        lineWidths = list(itertools.chain(*lineWidthsPerWriter)) # flatten array to get min and max later
        print 'Lines with width less than ', i_minAcceptableWidth, 'will be rejected'
        willBeNotRejected = [ _ for _ in itertools.compress(lineWidths, map(lambda x: x>=i_minAcceptableWidth, lineWidths)) ]
        minWidth = max(i_minAcceptableWidth, min(willBeNotRejected))
        
        # temp: cut line and take first third, just to train network TODO: take all info from line
        minWidth = minWidth / 3
    else:
        minWidth = i_specificWidth;
        
    if (i_specificHeight == -1):
        lineHeightsPerWriter = map(lambda x:map(lambda y:y.shape[0], x), linesToTrain.values())
        lineHeights = list(itertools.chain(*lineHeightsPerWriter)) # flatten array to get min and max later
        maxHeight = max(lineHeights)
    else:
        maxHeight = i_specificHeight;
    
    print 'Width of images -', minWidth
    print 'Height of images -', maxHeight
    
    varianceWriters = []
    # reject lines that shorter than minAcceptableWidth and pad others
    # and calculate variance of images in this loop
    for wId in linesToTrain.keys():
        linesForCurWriter = len(linesToTrain[wId])
        linesToTrain[wId] = [padWithOnesAndCut(l, maxHeight, minWidth) for l in linesToTrain[wId] if l.shape[1] > i_minAcceptableWidth]
        print 'Rejected', linesForCurWriter - len(linesToTrain[wId]), 'lines for writer ', wId, '( out of',linesForCurWriter,')'
        varianceWriters.append(np.var(linesToTrain[wId]))
    variance = np.mean(varianceWriters) # scalar - variance for all images
    
    print 'Preprocessing done'
    return (linesToTrain, minWidth, maxHeight, variance)
	
#####################################################################
def getLMDBEntry(i_image1, i_image2, i_label):
    datum = caffe.proto.caffe_pb2.Datum()
    #print 'sh' , i_image1.shape
    image = np.concatenate((i_image1, i_image2), axis=0) # top-im1, bot-im2
    datum.channels = 2
    datum.height = i_image1.shape[0]
    datum.width = i_image1.shape[1]
    
    binStr = image.tobytes() # binascii.hexlify
    datum.data = binStr
    
    #flatIm = np.fromstring(datum.data, dtype=np.uint8)
    #im = flatIm.reshape(datum.channels, datum.height, datum.width)
    
    datum.label = i_label
    return datum

#####################################################################    
def createLMDBpairs(i_nameLMDB, i_lines):
    map_size = 100000000000

    env = lmdb.open(i_nameLMDB, map_size=map_size)
    
    keys = i_lines.keys()
    indexLineLMDB = 0
    for i, wId in enumerate(i_lines):
    #    print 'writer', wId
        linesWi = i_lines[wId]
        for il, lineWi in enumerate(linesWi):
            
            # make pair of similar lines
            for iil in range(il + 1, len(linesWi)):
                lineWii = linesWi[iil] # another line from same writer
                # TODO move to func
                datum = getLMDBEntry(lineWi, lineWii, 1)
                with env.begin(write=True) as txn:
                    str_id = '{:08}'.format(indexLineLMDB)
                    txn.put(str_id.encode('ascii'), datum.SerializeToString()) # write to db
                indexLineLMDB = indexLineLMDB + 1
    #            print wId, ' ', wId, ': ', il, ' ', iil
                
            # make pair of different lines - take all from other writers           
            for wIdj in keys[i+1:]:
                linesWj = i_lines[wIdj] # lines of another author
                for jl, lineWj in enumerate(linesWj):
                    datum = getLMDBEntry(lineWi, lineWj, 0)
                    with env.begin(write=True) as txn:
                        str_id = '{:08}'.format(indexLineLMDB)
                        txn.put(str_id.encode('ascii'), datum.SerializeToString()) # write to db
                    indexLineLMDB = indexLineLMDB + 1
    #                print wId, ' ', wIdj, ': ', il, ' ', jl
    print '-> wrote ',indexLineLMDB, 'entried in LMDB'
    return
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

numWritersToTrain = 3
numLinesToTrain = 50
numLinesToTest = 20
# load forms, lines and words images for writers
# and create dict (idwriter1: all his lines, idwriter2: all his lines)
linesToTrain = {}
linesToTest = {}
for item in sortedWriters[1:numWritersToTrain+1]: # first wrote too much
    writer = item[1]
    linesToTrain[writer.id] = []
    linesToTest[writer.id] = []
    print 'Loading lines for writer ', writer.id, 'from', len(writer.formsRef), ' forms...'
    print 'Writer\'s total lines - ', writer.sumLines()
    for formId in writer.formsRef:
        for lineId in forms[formId].linesRef:
            lines[lineId].loadData()
            # for debug: take only few lines per writer
            if (len(linesToTrain[writer.id]) < numLinesToTrain):
                linesToTrain[writer.id].append(lines[lineId].data)
            elif (len(linesToTest[writer.id]) < numLinesToTest):
                linesToTest[writer.id].append(lines[lineId].data)
                
################################################################################
# preprocess images
minAcceptableWidth = 1000
(linesToTrain, lineWidth, lineHeight, varianceTrain) = preprocessImages(linesToTrain, minAcceptableWidth)
(linesToTest, lineWidth, lineHeight, varianceTest)  = preprocessImages(linesToTest, minAcceptableWidth, lineWidth, lineHeight)
#  len(linesToTrain.items()[0][1])
# plt.imshow(linesToTrain.items()[0][1][0])

################################################################################
#%% pack to pairs
print '------'
print 'Packing to LMDB pairs...'
print numLinesToTrain, 'lines for each writer will be used for creating training dataset'
print numLinesToTest, 'lines for each writer will be used for creating testing dataset'
dataFolder = "data"
nameLMDBtrain = os.path.join(dataFolder, 'pairs_train_lmdb')
createLMDBpairs(nameLMDBtrain, linesToTrain)
nameLMDBtest = os.path.join(dataFolder, 'pairs_test_lmdb')
createLMDBpairs(nameLMDBtest, linesToTest)

# to test - save one image
env = lmdb.open(nameLMDBtrain, readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000002')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flatIm = np.fromstring(datum.data, dtype=np.uint8)
#print flatIm.shape

im = flatIm.reshape(datum.channels, datum.height, datum.width)
#print im.shape
#plt.imshow(im)

import scipy
scipy.misc.imsave('network/testPair.jpg', im[1, :, :]) # im in second channel

print 'Packing finished'

print '=============================='
print '!! Before new training, dont dorget to:'
print '1. Recalculate train and test mean'
print '2. In train_test prototxt:'
print '                scale for train - ', 1 / varianceTrain
print '                scale for test - ', 1 / varianceTest
print '!! Before predicting, dont dorget to:'
print 'In deploy prototxt file:'
print '                input_dim height', lineHeight
print '                input_dim width', lineWidth
