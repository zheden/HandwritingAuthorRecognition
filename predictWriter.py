import caffe
import leveldb
import numpy as np
import scipy
import sys
import os
import Image
import operator
from collections import Counter
from nearestNeighbor import startMain

caffe.set_mode_gpu()


###################################################################################
def extractDescriptor(im, net):
    net.blobs['data'].data[...] = im
    out = net.forward()

    outD1 = net.blobs['descriptor'].data[0]
    return outD1.tolist()
    
###################################################################################
def getDescriptorsTest(i_images, net):
    print 'Getting descriptors for test data ...'

    descriptorsOut = []
    ii = 0
    for im in i_images:
        desc = extractDescriptor(im, net)
        descriptorsOut.append(desc)
        ii += 1
        if ii % 1000 == 0:
            print '  got', ii, 'descriptors'

    return descriptorsOut

###################################################################################
def loadImages(images_path, mean_file):
    print 'Loading images from', images_path, '...'
    imMean = np.fromfile(mean_file, dtype=np.float32)
    imMean = imMean.reshape((35, 100))

    listingWriters = os.listdir(images_path)  

    train_images = []
    train_labels = []
    ii = 0
    for folderW in listingWriters:
        path = images_path + '/' + folderW
        listingImages = os.listdir(path)
        writerId = int(folderW)
        for imageName in listingImages:
            im = Image.open(path + '/' + imageName)
            im = np.array(im.resize((100, 35)))
            im = im / 255.0 - imMean
            train_images.append(im)
            train_labels.append(writerId)
            ii += 1
            if ii % 1000 == 0:
                print '  loaded', ii, 'images'

    return (train_images, train_labels)

###################################################################################
def getDescriptorSpace(train_images, train_labels, net):
    print 'Getting descriptors for train data...'
    train_descriptors = {}
    train_descriptors_flat = []
    writerIds = set(train_labels)
    for wId in writerIds:
        train_descriptors[wId] = []
    
    counter = 0
    for im in train_images:
        desc = extractDescriptor(im, net)
        train_descriptors[train_labels[counter]].append(desc)
        train_descriptors_flat.append(desc)
        counter += 1
    print 'done'
    return (train_descriptors, train_descriptors_flat)
    
###################################################################################
def most_common(lst):
    if (len(lst) == 1):
        return lst[0]
    data = Counter(lst)
    if (data.most_common(1)[0][1] == 1):
        # when all neigh are different
        return -1 #lst[0]
    else:
        return data.most_common(1)[0][0]

###################################################################################
def doKNN(i_trainDesc, i_trainLabels, i_testDesc, k1, k2, is_validation = True):
    testDesc = np.array(i_testDesc)
    distances = [sum(np.power(x - testDesc, 2)) for x in i_trainDesc]
    indShift = 0
    if (not is_validation):
        indShift = 1 # if it is train data, dont take first, because it will be desc itself
        
    distancesSorted, labelsSorted = zip(*sorted(zip(distances, i_trainLabels)))
    labelK1 = most_common(labelsSorted[0 + indShift : k1 + indShift])
    labelK2 = most_common(labelsSorted[0 + indShift : k2 + indShift])
    return (labelK1, labelK2)
    
    #if (k > 1 or (k == 1 and indShift > 0)):
    #    distancesSorted, labelsSorted = zip(*sorted(zip(distances, i_trainLabels)))
    #    return most_common(labelsSorted[0 + indShift : k + indShift])
    #else:
    #    ind = distances.index(min(distances))
    #    return i_trainLabels[ind]

###################################################################################
def predict(train_descriptors_flat, train_labels, test_descriptors, k1, k2, is_validation = True):
    pred_labelsK1 = []
    pred_labelsK2 = []
    ii = 0
    for desc in test_descriptors:
        desc = np.array(desc) # TODO remove
        (labelK1, labelK2) = doKNN(train_descriptors_flat, train_labels, desc, k1, k2, is_validation)
        pred_labelsK1.append(labelK1)
        pred_labelsK2.append(labelK2)
        ii += 1
        if ii % 1000 == 0:
            print '  predicted', ii, 'writers'
    return (pred_labelsK1, pred_labelsK2)
###################################################################################
###################################################################################
###################################################################################

def runPrediction(PRETRAINDED_MODEL_FILE, MODEL_PROTO_FILE, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, MEAN_FILE):
    # used model
    net = caffe.Classifier(MODEL_PROTO_FILE, PRETRAINDED_MODEL_FILE)
    print '---------------------------------------------------------------------------------------------------------'
    ###################### get train info
    #### create descriptor space
    (train_images, train_labels) = loadImages(TRAIN_IMAGES_PATH, MEAN_FILE)
    (train_descriptors, train_descriptors_flat) = getDescriptorSpace(train_images, train_labels, net)

    '''
    train_descriptors = np.load('data/train_descriptors.npy').item()
    '''

    ###################### get test info
    test_images_npy_file = TEST_IMAGES_PATH + 'test_images.npy'
    test_labels_npy_file = TEST_IMAGES_PATH + 'test_labels.npy'
    if os.path.isfile(test_images_npy_file):
        test_images = np.load(test_images_npy_file)
        test_labels = np.load(test_labels_npy_file)
        print 'Loaded test images from', test_images_npy_file
    else:
        (test_images, test_labels) = loadImages(TEST_IMAGES_PATH, MEAN_FILE)
        np.save(test_images_npy_file, test_images)
        np.save(test_labels_npy_file, test_labels)

    # predict with 1NN and 3NN
    k1 = 1
    k3 = 3

    print ''
    print '------------------------- PREDICT TRAIN------------------------------------------------------------------'
    is_validation = False
    (predicted_labels_trK1, predicted_labels_trK3) = predict(train_descriptors_flat, train_labels, train_descriptors_flat, k1, k3, is_validation)
    
    prediction_rate_trK1 = 1.0 * sum(np.equal(predicted_labels_trK1, train_labels)) / len(train_labels)
    prediction_rate_trK3 = 1.0 * sum(np.equal(predicted_labels_trK3, train_labels)) / len(train_labels)
    print '*************************** TRAIN DATA: PREDICTION RATE 1NN = ', prediction_rate_trK1
    print '*************************** TRAIN DATA: PREDICTION RATE 3NN = ', prediction_rate_trK3


    print ''
    print '------------------------- PREDICT VAL------------------------------------------------------------------'
    test_descriptors = getDescriptorsTest(test_images, net)
    is_validation = True
    (predicted_labels_valK1, predicted_labels_valK3) = predict(train_descriptors_flat, train_labels, test_descriptors, k1, k3, is_validation)
    
    prediction_rate_valK1 = 1.0 * sum(np.equal(predicted_labels_valK1, test_labels)) / len(test_labels)
    prediction_rate_valK3 = 1.0 * sum(np.equal(predicted_labels_valK3, test_labels)) / len(test_labels)
    print '*************************** VALIDATION DATA: PREDICTION RATE 1NN = ', prediction_rate_valK1
    print '*************************** VALIDATION DATA: PREDICTION RATE 3NN = ', prediction_rate_valK3


    '''
    #################################### to check old  - test on train data
    print ''
    print '------------------------- TRY ONE ON TRAIN DATA ------------------------------------------------------------------'
    # tt - from train dataset
    # after startMain we get id1, id2, id3 which is wrong because there are close descriptors from id 1.
    # after doKNN we get id1, id1, id1
    tt = [train_descriptors_flat[0]]


    testDesc = np.array(tt[0])
    is_validation = True
    print doKNN(train_descriptors_flat, train_labels, testDesc, k1, k3, is_validation)

    print '----------------------kNN from nearestNeighbor.py------------------------------------------------------------------'
    predicted_labelsT = startMain(train_descriptors, tt, k3)
    '''  
    return (prediction_rate_trK1, prediction_rate_trK3, prediction_rate_valK1, prediction_rate_valK3)

###################################################################################
###################################################################################
###################################################################################
if __name__ == "__main__":
    TRAIN_IMAGES_PATH = 'data/train_images_14/'
    MEAN_FILE = 'data/train_mean.bin'
    TEST_IMAGES_PATH = 'data/test_images/'

    PRETRAINDED_MODEL_FILE = "network/snapTriplet/evg_writer_triplet_iter_4000.caffemodel"
    #PRETRAINDED_MODEL_FILE = "network/george_500.caffemodel"
    MODEL_PROTO_FILE = "network/evg_writer_online_1.prototxt"
    #MODEL_PROTO_FILE = "network/george_writer_online_1.prototxt"
    
    runPrediction(PRETRAINDED_MODEL_FILE, MODEL_PROTO_FILE, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, MEAN_FILE)


