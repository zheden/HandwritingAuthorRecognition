import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import caffe
import lmdb
import scipy
from PIL import Image
import itertools
import shutil
import random

DBpath = os.path.join(".", 'IAM')
formsPath = os.path.join(DBpath, 'forms')
linesPath = os.path.join(DBpath, 'lines')
wordsPath = os.path.join(DBpath, 'words')
asciiPath = os.path.join(DBpath, 'ascii')
formsFile = os.path.join(asciiPath, 'forms.txt')
linesFile = os.path.join(asciiPath, 'lines.txt')
wordsFile = os.path.join(asciiPath, 'words.txt')
newLinesPath = os.path.join(DBpath, 'new lines')

#####################################################################
class Writer:
    
    def __init__(self, i_id = -1):
        self.id = i_id
        self.formsRef = []
        self.savedSumLines = -1;
        self.savedSumWords = -1;
        
    def __repr__(self):
        return ('Writer (id=%s, forms=%s, lines=%s, words=%s)'
                % (repr(self.id), repr(len(self.formsRef)), repr(self.savedSumLines), repr(self.savedSumWords) ))

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

    def __init__(self, i_id = '', i_label = '', i_boundingBox = ()):
        self.id = i_id
        self.wordsRef = []
        self.label = i_label
        self.data = []
        self.boundingBox = i_boundingBox

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
    def __init__(self, i_id = '', i_label = '', i_boundingBox = ()):
        self.id = i_id
        self.label = i_label
        self.data = []
        self.boundingBox = i_boundingBox

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
def padWithOnesOrCut(line, i_maxHeight):
    curHeight = line.shape[0]
    if (curHeight <= i_maxHeight):
        # pad on top and bottom
        padOnTop = (i_maxHeight - curHeight) / 2
        padOnBot = i_maxHeight - curHeight - padOnTop
        line = np.pad(line, ((padOnTop, padOnBot), (0, 0)), mode='constant', constant_values=(255))[:, :]
    else:
        # cut on top and bottom
        d = (curHeight - i_maxHeight) / 2;
        if (d > 0):
            line = line[d:-d, :]
        if (line.shape[0] <> i_maxHeight): # can be larger 1 pix
            line = line[0:-1, :]
    line = np.array(255 * np.ones(line.shape) - line, dtype=np.uint8) # invert
    return line

#####################################################################
# input and output : dict (idwriter1: all his lines, idwriter2: all his lines)
def preprocessImages(linesToTrain, i_specificHeight = -1):
    print 'Preprocessing images for', len(linesToTrain), 'writers:'
        
    if (i_specificHeight == -1):
        lineHeightsPerWriter = map(lambda x:map(lambda y:y.shape[0], x), linesToTrain.values())
        lineHeights = list(itertools.chain(*lineHeightsPerWriter)) # flatten array to get min and max later
        maxHeight = sum(lineHeights) / len(lineHeights) # mean height
    else:
        maxHeight = i_specificHeight;
    
    print 'Height of images -', maxHeight
    
    varianceWriters = []
    # calculate variance of images TODO: remove if will not use
    for wId in linesToTrain.keys():
        linesToTrain[wId] = [padWithOnesOrCut(l, maxHeight) for l in linesToTrain[wId]]
        varianceWriters.append(np.var(linesToTrain[wId]))
    variance = np.mean(varianceWriters) # scalar - variance for all images
    
    print 'Preprocessing done'
    return (linesToTrain, maxHeight, variance)


def saveToDisk(wordGroupId, img, word_count, lineSample, writerSample):
    # Saving details to a text file
    with open(asciiPath + '/new lines.txt', 'a') as file:
        file.write("[" + str(word_count) + "] " + lineSample + '-' + str(wordGroupId) + '-' + str(writerSample[0]) + '\n')

    # OPTIONAL..!!! Saving new line images to disk

    img = img.convert('RGB')
    directory = newLinesPath + '/' + str(writerSample[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    newFilePath = directory + '/' + lineSample + '_' + str(wordGroupId) + '.png'
    img.save(newFilePath) # save to disk

#####################################################################

def getLMDBEntryPair(i_image1, i_image2, i_label):
    datum = caffe.proto.caffe_pb2.Datum()
    image = np.dstack((i_image1, i_image2)) # ch1 -im1, ch2 -im2
    
    image = image.swapaxes(0, 2).swapaxes(1, 2) # (ch, h, w)
    
    datum.channels = 2
    datum.height = i_image1.shape[0]
    datum.width = i_image1.shape[1]
    
    binStr = image.tobytes()
    datum.data = binStr
    
    #flatIm = np.fromstring(datum.data, dtype=np.uint8)
    #im = flatIm.reshape(datum.height, datum.width, datum.channels)
    
    datum.label = i_label
    return datum

#####################################################################
def getLMDBEntryTriplet(i_image1, i_image2, i_image3):
    datum = caffe.proto.caffe_pb2.Datum()
    image = np.dstack((i_image1, i_image2, i_image3)) # ch1 -im1, ch2 -im2, ch3 -im3
    image = image.swapaxes(0, 2).swapaxes(1, 2) # (ch, h, w)
    
    datum.channels = 3
    datum.height = i_image1.shape[0]
    datum.width = i_image1.shape[1]
    
    binStr = image.tobytes() 
    datum.data = binStr
    
    return datum
    
#####################################################################    
def createLMDBpairs(i_nameLMDB, i_lines):
    map_size = 10000000000000 # TODO use deepdish instead of this ugly num http://deepdish.io/2015/04/28/creating-lmdb-in-python/

    shutil.rmtree(i_nameLMDB, True)
    env = lmdb.open(i_nameLMDB, map_size=map_size)
    
    keys = i_lines.keys()
    numWriters = len(keys)
    indexLineLMDB = 0
    for i, wId in enumerate(i_lines):
    #    print 'writer', wId
        linesWi = i_lines[wId]
        for il, lineWi in enumerate(linesWi):
            
            # make pair of similar lines
            numLinesWi = len(linesWi)
            for iil in range(il + 1, numLinesWi):
                lineWii = linesWi[iil] # another line from same writer
                datum = getLMDBEntryPair(lineWi, lineWii, 1)
                with env.begin(write=True) as txn:
                    str_id = '{:08}'.format(indexLineLMDB)
                    txn.put(str_id.encode('ascii'), datum.SerializeToString()) # write to db
                indexLineLMDB = indexLineLMDB + 1
    #            print wId, ' ', wId, ': ', il, ' ', iil
                
            # make pair of different lines - take all from other writers           
            for wIdj in keys[i+1:]:
                counterLinesWj = 0
                linesWj = i_lines[wIdj] # lines of another author
                for jl, lineWj in enumerate(linesWj):
                    counterLinesWj = counterLinesWj + 1
                    if (counterLinesWj > numLinesWi / numWriters): # to have ~ equal num of 0 and 1 labels
                        break
                    datum = getLMDBEntryPair(lineWi, lineWj, 0)
                    with env.begin(write=True) as txn:
                        str_id = '{:08}'.format(indexLineLMDB)
                        txn.put(str_id.encode('ascii'), datum.SerializeToString()) # write to db
                    indexLineLMDB = indexLineLMDB + 1
    #                print wId, ' ', wIdj, ': ', il, ' ', jl
    print '-> wrote ',indexLineLMDB, 'entried in LMDB'
    env.close()
    return
    
#####################################################################    
# all to all
def createLMDBtriplets(i_nameLMDB, i_lines):
    map_size = 10000000000000

    shutil.rmtree(i_nameLMDB, True)
    env = lmdb.open(i_nameLMDB, map_size=map_size)
    
    keys = i_lines.keys()
    indexLineLMDB = 0
    for i, wId in enumerate(i_lines):
        linesWi = i_lines[wId]  # TODO: instead of selecting line, call func that combines random words of that writer
        # loop through lines of a writer
        for il, lineWi in enumerate(linesWi):
            
            # loop through lines of same writer, starting from next
            for iil in range(il + 1, len(linesWi)):
                lineWii = linesWi[iil] # another line from same writer # TODO: instead of selecting line, call func that combines random words of that writer
                
                # loop through lines of all other writers      
                for wIdj in keys[i+1:]:
                    linesWj = i_lines[wIdj] # lines of another author # TODO: instead of selecting line, call func that combines random words of that writer
                    for jl, lineWj in enumerate(linesWj):
                        datum = getLMDBEntryTriplet(lineWi, lineWii, lineWj)
                        with env.begin(write=True) as txn:
                            str_id = '{:08}'.format(indexLineLMDB)
                            txn.put(str_id.encode('ascii'), datum.SerializeToString()) # write to db
                        indexLineLMDB = indexLineLMDB + 1
    print '-> wrote ',indexLineLMDB, 'entried in LMDB'
    env.close()
    return
    
#####################################################################
def applyPermutations(i_wordsInLine, MAX_WIDTH, i_lineBoundingBox, i_lineSample, i_writerSample, i_wordsSpace):
    combinationSeen = [] # to keep track of combinations of words (for a line) seen so far
    wordGroupId = -1
    outputLines = []
    for wordGroup in itertools.permutations(i_wordsInLine, len(i_wordsInLine)): # iterate for each combination of words
        total_width = 0 # to keep track of total width of line formed so far, so that it doesn't exceed MAX_WIDTH
        word_count = 0 # to count number of words considered to be included in line
        wordsSet = [] # final set of words which form a new line. Remaining words are not considered because of limited MAX_WIDTH
        for word in wordGroup:
            #print total_width
            if total_width > MAX_WIDTH: # we don't add words to wordsSet because MAX_WIDTH has been achieved
                # if wordsSet already resides in combinationSeen then don't consider new line formation to avoid duplicate new lines
                if wordsSet in combinationSeen:
                    wordsSet = []
                    break
                else:
                    combinationSeen.append(wordsSet)
                    wordGroupId += 1
                    #print wordGroup
                    # initializing new bounding box for line in which words will be pasted
                    outerBox = np.empty((i_lineBoundingBox[3], MAX_WIDTH))
                    outerBox.fill(255)
                    imgLine = Image.fromarray(outerBox) # creating empty image from bounding box array
                    xPixelsCovered = 0 # to set x position of new word added to the line
                    for word in wordsSet:
                        #print word
                        yStartPos = words[word].boundingBox[1]-i_lineBoundingBox[1] # calculate a word's starting position on Y axis in bounding box
                        wordImg = Image.fromarray(words[word].data) # Load word's image data
                        imgLine.paste(wordImg, (xPixelsCovered,yStartPos))
                        xPixelsCovered += wordImg.size[0] + i_wordsSpace # i_wordsSpace extra pixels added to create emtpy space between words
                        #print xPixelsCovered
                    #imgLine.show()
                    #saveToDisk(wordGroupId, imgLine, word_count, i_lineSample, i_writerSample)
                    outputLines.append(np.array(imgLine))
                    break
            wordsSet.append(word)
            wordWidth = words[word].boundingBox[2]
            total_width += wordWidth+(word_count*i_wordsSpace) # update total_width covered so far
            word_count += 1
    return outputLines
#####################################################################
#####################################################################

print '----------------------------------------------------------------------------'
print '----------------------------------------------------------------------------'  
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
    boundingBoxX = int(columns[4])
    boundingBoxY = int(columns[5])
    boundingBoxWidth = int(columns[6])
    boundingBoxHeight = int(columns[7])

    lines[lineId] = Line(lineId, label, (boundingBoxX, boundingBoxY, boundingBoxWidth, boundingBoxHeight))

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
    boundingBoxX = int(columns[3])
    boundingBoxY = int(columns[4])
    boundingBoxWidth = int(columns[5])
    boundingBoxHeight = int(columns[6])

    words[wordId] = Word(wordId, label, (boundingBoxX, boundingBoxY, boundingBoxWidth, boundingBoxHeight))

    #print 'wordId : ' + wordId

    # add this word to corresponding line
    lineId = wordId[0:-3]
    lines[lineId].wordsRef.append(wordId)
f.close()

#%% save how num of Lines and words
for writerId in writers:
    writers[writerId].calculateAnsSaveSums()

#%%
sortedWriters = sorted(writers.items(), key=lambda w: w[1].savedSumWords, reverse=True)

############################ Concatenating words to form lines ######################################
MAX_WRITERS = 3
# Maximum allowed width of lines to be formed, height is mean value for all heights
MAX_WIDTH = 200

MAX_LINES_TO_TRAIN = 40 # NOTE: this is number without permutations
MAX_LINES_TO_TEST = 2
MAX_NUM_WORDS_PERMUTATIONS = 4

linesToTrain = {} # dict (idwriter1: all his lines, idwriter2: all his lines)
linesToTest = {}

numLinesToTrain = 0 # counters how many lines we created for all writers
numLinesToTest = 0

print '--------num writers -', MAX_WRITERS
print '--------num lines train per writer -', MAX_LINES_TO_TRAIN
print '--------num lines test per writer -', MAX_LINES_TO_TEST
print '--------max permutations per line -', MAX_NUM_WORDS_PERMUTATIONS

writersCount = 0
for writerSample in sortedWriters:
    if writerSample[0] == 0: # Ignore first writer who wrote too much
        continue
    writersCount += 1
    if writersCount > MAX_WRITERS:
        break
    writer = writerSample[1]
    linesToTrain[writer.id] = []
    linesToTest[writer.id] = []
    print 'Loading lines for', writer
    writerForms = writer.formsRef
    #print writerForms
    formsCount = 0
    linesCount = 0 # total number of lines, not per form
    for formSample in writerForms:
        #print forms[formSample]
        formLines = forms[formSample].linesRef
        for lineSample in formLines:
            if linesCount >= (MAX_LINES_TO_TRAIN + MAX_LINES_TO_TEST):
                break
            print lines[lineSample]
            wordsInLine = lines[lineSample].wordsRef
            lineBoundingBox = lines[lineSample].boundingBox
            for wordId in wordsInLine: # load images for words of the line
                words[wordId].loadData()
            # remove short words like 'a', '.'
            wordsInLine = [wId for wId in wordsInLine if (words[wId].data.shape[1] > 25)]
            
            # TEMP
            numToTake = min(8, len(wordsInLine))
            wordsInLine = wordsInLine[0:numToTake]
            
            
            # do permutations of words
            print 'do permutations for line ', linesCount, 'out of', MAX_LINES_TO_TRAIN + MAX_LINES_TO_TEST, '...'
            linesWithWordPermutations = applyPermutations(wordsInLine, MAX_WIDTH, lineBoundingBox, lineSample, writerSample, 10)
            if (MAX_NUM_WORDS_PERMUTATIONS < len(linesWithWordPermutations)):
                linesWithWordPermutations = random.sample(linesWithWordPermutations, MAX_NUM_WORDS_PERMUTATIONS)

            if (linesCount < MAX_LINES_TO_TRAIN):
                # write train set
                linesToTrain[writer.id] += linesWithWordPermutations
            else:
                # write test set. TODO: do not need permutations here?
                linesToTest[writer.id] += linesWithWordPermutations
            linesCount += 1
        # end loop for lines
    # end loop for forms
    if (linesCount > MAX_LINES_TO_TRAIN):
        numLinesToTrainForCurWriter = MAX_LINES_TO_TRAIN
        numLinesToTestForCurWriter = linesCount - MAX_LINES_TO_TRAIN
    else:
        numLinesToTrainForCurWriter = linesCount
        numLinesToTestForCurWriter = 0
    
    print '    loaded', numLinesToTrainForCurWriter, 'lines for training', ', in total with permutations - ', len(linesToTrain[writer.id])
    print '    loaded', numLinesToTestForCurWriter, 'lines for testing', ', in total with permutations - ', len(linesToTest[writer.id])
    numLinesToTrain += len(linesToTrain[writer.id])
    numLinesToTest += len(linesToTest[writer.id])
# end loop for writers

###########################################################################

# preprocess images
print '------ Train set:'
(linesToTrain, lineHeight, varianceTrain) = preprocessImages(linesToTrain)
print '------ Test set:'
(linesToTest, lineHeight, varianceTest)  = preprocessImages(linesToTest, lineHeight)
#  len(linesToTrain.items()[0][1])
# plt.imshow(linesToTrain.items()[0][1][0])

################################################################################
#%% pack to pairs
packToTriplets = False
dataFolder = "data"
print '----------------------------------------------------------------------------'
print numLinesToTrain, 'lines in total will be used for creating training dataset'
print numLinesToTest, 'lines in total will be used for creating testing dataset'
if (not packToTriplets):
    print 'Packing to LMDB pairs...'
    nameLMDBtrain = os.path.join(dataFolder, 'pairs_train_lmdb')
    createLMDBpairs(nameLMDBtrain, linesToTrain)
    nameLMDBtest = os.path.join(dataFolder, 'pairs_test_lmdb')
    createLMDBpairs(nameLMDBtest, linesToTest)
else:
    print 'Packing to LMDB triplets...'
    nameLMDBtrain = os.path.join(dataFolder, 'triplets_train_lmdb')
    createLMDBtriplets(nameLMDBtrain, linesToTrain)
    nameLMDBtest = os.path.join(dataFolder, 'triplets_test_lmdb')
    createLMDBtriplets(nameLMDBtest, linesToTest)

################################################################################
# to test - save one image
env = lmdb.open(nameLMDBtrain, readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000002')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flatIm = np.fromstring(datum.data, dtype=np.uint8)
im = flatIm.reshape(datum.channels, datum.height, datum.width)

scipy.misc.imsave('testPair_from_trainLMDB.jpg', im[1, :, :]) # im in second channel

print 'Packing finished'

print '----------------------------------------------------------------------------'
print '----------------------------------------------------------------------------'
print '!! Before new training, dont forget to:'
print '   * Recalculate train and test mean'
print '!! Before predicting, dont forget to:'
print '   * In deploy prototxt file:'
print '                input_dim height', lineHeight
print '                input_dim width', MAX_WIDTH
