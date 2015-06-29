
import caffe
import lmdb
import leveldb
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

from nearestNeighbor import startMain

USE_LMDB = True
descriptors = {}
RUN_COUNT = 5000

def extractDescriptors(i_raw_datum, net, i_transformer):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(i_raw_datum)

    im = caffe.io.datum_to_array(datum)
    #scipy.misc.imsave('testIm1.jpg', im[0, :, :])
    #scipy.misc.imsave('testIm2.jpg', im[1, :, :])
    #scipy.misc.imsave('testIm3.jpg', im[2, :, :])

    caffe.set_mode_cpu()
    if (USE_LMDB):
        imS0 = im
        imS0 = np.swapaxes(im, 1, 0) # TODO WHY?
        imS0 = np.swapaxes(imS0, 1, 2)
        preprocessedData = i_transformer.preprocess('data', imS0)
        #print 'preprocessed'
    else:
        # in leveldb George olready subtracted mean and scaled
        imS0 = im
        preprocessedData = imS0

    net.blobs['triplet_data'].data[...] = preprocessedData
    #net.blobs['sim'].data[...] = datum.label
    #scipy.misc.imsave('inputData0.jpg', net.blobs['triplet_data'].data[0][0])
    out = net.forward()

    outD1 = net.blobs['descriptor_1'].data[0]
    #outD2 = net.blobs['descriptor_2'].data[0]
    #outD3 = net.blobs['descriptor_3'].data[0]
    #print '----------------------------'
    #print outD1

    writerId = datum.label
    if writerId in descriptors:
        if outD1.tolist() not in descriptors[writerId]:
            descriptors[writerId].append(outD1.tolist())
    else:
        descriptors[writerId] = [outD1.tolist()]
    return descriptors

def runPrediction():
    print '-------------------------------------------------------------------'
    print '-------------------------------------------------------------------'
    ## get image from testing dataset

    if (USE_LMDB):
        pretrainedFile = "network/snapTriplet/snap_triplet_author_rec_iter_200.caffemodel"
        env = lmdb.open("data/triplets_train_lmdb", readonly=True)
        modelFile = "network/triplets_deploy_final.prototxt"

        ###################################################################################
        # set model
        sys.argv = ['', 'data/triplets_train_mean.binaryproto', 'data/triplets_train_mean.npy']
        execfile("data/convertMean.py")
        meanTest = np.load("data/triplets_train_mean.npy")
        net = caffe.Classifier(modelFile, pretrainedFile,
                               #mean = meanTest #   [scalar for 1 channel]
                               #raw_scale=255,
                               #image_dims=(38, 106)
                               )

        ###################################################################################

        # subtract mean and scale. In Lmdb original images are stored, mean subtraction and scaling was done with network during training
        transformer = caffe.io.Transformer({'data': net.blobs['triplet_data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', meanTest) # mean pixel
        transformer.set_input_scale('data', 0.00390625)  # the reference model operates on images in [0,255] range instead of [0,1]
        #transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        with env.begin() as txn:
            test_descriptors = [[0.99, -0.16, 0.14, 0.85, 0.44, -0.22, 0.64, -0.55, -0.01, -1.06, 1.03, -0.45, 0.08, -0.88, 0.14, 0.11]]
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                raw_datum = value
                train_descriptors = extractDescriptors(raw_datum, net, transformer)
                count += 1
                if count % 1000 == 0:
                    print count
                #if count > RUN_COUNT:
                 #   break
        return train_descriptors, test_descriptors
    else:
        pretrainedFile = "network/snap/leveldb_snap_triplet_author_rec_iter_500.caffemodel"
        modelFile = "network/leveldb_triplets_deploy.prototxt"

        ###################################################################################
        # set model
        sys.argv = ['', 'data/triplets_train_mean.binaryproto', 'data/triplets_train_mean.npy']
        execfile("data/convertMean.py")
        meanTest = np.load("data/triplets_train_mean.npy")
        net = caffe.Classifier(modelFile, pretrainedFile,
                               #mean = meanTest #   [scalar for 1 channel]
                               #raw_scale=255,
                               #image_dims=(38, 106)
                               )
        ###################################################################################

        db = leveldb.LevelDB('./data/train_db')
        raw_datum = db.Get('pos_5')
        train_descriptors = extractDescriptors(raw_datum, net, transformer='')
        return train_descriptors

if __name__ == "__main__":
    train_descriptors, test_descriptors = runPrediction()
    print train_descriptors
    #print '-----------------------------------------------------'
    #print test_descriptors
    startMain(train_descriptors, test_descriptors)