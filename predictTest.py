
import caffe
import lmdb
import leveldb
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(i_data, filename, padsize=1, padval=0):
    data = i_data - i_data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data[:, :, 0])
    dataToSave = scipy.ndimage.zoom(data[:, :, 0], 7, order=0)
    scipy.misc.imsave(filename, dataToSave)

print '-------------------------------------------------------------------'
print '-------------------------------------------------------------------'
## get image from testing dataset

USE_LMDB = True
if (USE_LMDB):
    pretrainedFile = "network/snapTriplet/snap_triplet_author_rec_iter_10.caffemodel"
    env = lmdb.open("data/triplets_train_lmdb", readonly=True)
    modelFile = "network/triplets_deploy.prototxt"
    with env.begin() as txn:
        raw_datum = txn.get(b'00000022')
else:
    pretrainedFile = "network/snap/snap_triplet_author_rec_iter_10.caffemodel"
    modelFile = "network/leveldb_triplets_deploy.prototxt"
    db = leveldb.LevelDB('./data/train_db')
    raw_datum = db.Get('t5')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

im = caffe.io.datum_to_array(datum)
scipy.misc.imsave('testIm1.jpg', im[0, :, :])
scipy.misc.imsave('testIm2.jpg', im[1, :, :])
scipy.misc.imsave('testIm3.jpg', im[2, :, :])


caffe.set_mode_cpu()
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
print '\n------------------ analyse outputs from each layer ----------------------------'
# save conv masks
# the parameters are a list of [weights, biases]
filters1 = net.params['conv1'][0].data
ff1 = filters1.transpose(0, 2, 3, 1)
ff1im = vis_square(ff1, "filters1.jpg")

filters2 = net.params['conv2'][0].data
ff2 = filters2.transpose(0, 2, 3, 1)
ff2im = vis_square(ff2, "filters2.jpg")


if (USE_LMDB):
    imS0 = im
    imS0 = np.swapaxes(im, 1, 0) # TODO WHY?
    imS0 = np.swapaxes(imS0, 1, 2)

    # subtract mean and scale. In Lmdb original images are stored, mean subtraction and scaling was done with network during training
    transformer = caffe.io.Transformer({'data': net.blobs['triplet_data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', meanTest) # mean pixel
    transformer.set_input_scale('data', 0.00390625)  # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    preprocessedData = transformer.preprocess('data', imS0)
else:
    # in leveldb George olready subtracted mean and scaled
    imS0 = im
    preprocessedData = imS0

net.blobs['triplet_data'].data[...] = preprocessedData
net.blobs['label'].data[...] = datum.label
scipy.misc.imsave('inputData0.jpg', net.blobs['triplet_data'].data[0][0])
out = net.forward()

outputSlicer1 = net.blobs['data_1'].data
outputSlicer2 = net.blobs['data_2'].data
outputSlicer3 = net.blobs['data_3'].data


print 'Read from DB:   shape', imS0.shape, 'mean', imS0.mean(), ', min' ,imS0.min(), ', max', imS0.max()
print 'Ouput slicer:   shape', outputSlicer1.shape, 'mean', outputSlicer1.mean(), ', min' ,outputSlicer1.min(), ', max', outputSlicer1.max()
print 'Ouput slicer2:  shape', outputSlicer2.shape, 'mean', outputSlicer2.mean(), ', min' ,outputSlicer2.min(), ', max', outputSlicer2.max()
print 'Ouput slicer3:  shape', outputSlicer3.shape, 'mean', outputSlicer3.mean(), ', min' ,outputSlicer3.min(), ', max', outputSlicer3.max()

################ conv
print ''
outConv1_1 = net.blobs['conv1_1'].data
outConv1_2 = net.blobs['conv1_2'].data
outConv1_3 = net.blobs['conv1_3'].data
print 'Output conv1_1 Min, max' , outConv1_1.min(), ' ', outConv1_1.max()
print 'Output conv1_2 Min, max' , outConv1_2.min(), ' ', outConv1_2.max()
print 'Output conv1_3 Min, max' , outConv1_3.min(), ' ', outConv1_3.max()

print ''
outConv2_1 = net.blobs['conv2_1'].data
outConv2_2 = net.blobs['conv2_2'].data
outConv2_3 = net.blobs['conv2_3'].data
print 'Output conv2_1 Min, max' , outConv2_1.min(), ' ', outConv2_1.max()
print 'Output conv2_2 Min, max' , outConv2_2.min(), ' ', outConv2_2.max()
print 'Output conv2_3 Min, max' , outConv2_3.min(), ' ', outConv2_3.max()


############## fc
print ''
wFc1_1 = net.params['fc1_1'][0].data
print 'Weights for fc1, mean', wFc1_1.mean(), ', min' ,wFc1_1.min(), ', max', wFc1_1.max()
bFc1_1 = net.params['fc1_1'][1].data

outFc1_1 = net.blobs['fc1_1'].data[0]
outFc1_2 = net.blobs['fc1_2'].data[0]
outFc1_3 = net.blobs['fc1_3'].data[0]
print 'Output fc1_1: mean', outFc1_1.mean(), ', min' ,outFc1_1.min(), ', max', outFc1_1.max()
print 'Output fc1_2: mean', outFc1_2.mean(), ', min' ,outFc1_2.min(), ', max', outFc1_2.max()
print 'Output fc1_3: mean', outFc1_3.mean(), ', min' ,outFc1_3.min(), ', max', outFc1_3.max()


print ''
wD1 = net.params['descriptor_1'][0].data
print 'Weights for descriptor:   mean', wD1.mean(), ', min' ,wD1.min(), ', max', wD1.max()

outD1 = net.blobs['descriptor_1'].data[0]
outD2 = net.blobs['descriptor_2'].data[0]
outD3 = net.blobs['descriptor_3'].data[0]
print ''
print 'Output descriptor_1:   mean', outD1.mean(), ', min' ,outD1.min(), ', max', outD1.max()
print 'Output descriptor_2:   mean', outD2.mean(), ', min' ,outD2.min(), ', max', outD2.max()
print 'Output descriptor_3:   mean', outD3.mean(), ', min' ,outD3.min(), ', max', outD3.max()

distSimilar = np.linalg.norm(outD1-outD2)
distDissimilar13 = np.linalg.norm(outD1-outD3)
print 'distance betw descriptors:  similar - ', distSimilar, 'dissimilar - ', distDissimilar13

lossT = net.blobs['lossTriplet'].data
lossP = net.blobs['lossPair'].data
print 'Loss triplet for this image', lossT
print 'Loss pair for this image', lossP
