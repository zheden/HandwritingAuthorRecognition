
import caffe
import lmdb
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, filename, padsize=1, padval=0):
    data -= data.min()
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
    return data

print '-------------------------------------------------------------------'
print '-------------------------------------------------------------------'
# get image from testing dataset
env = lmdb.open("data/pairs_train_lmdb", readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000') # diff authors here

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flatIm = np.fromstring(datum.data, dtype=np.uint8)
im = flatIm.reshape(datum.channels, datum.height, datum.width)
scipy.misc.imsave('testIm1.jpg', im[0, :, :])
scipy.misc.imsave('testIm2.jpg', im[1, :, :])


caffe.set_mode_cpu()
pretrainedFile = "network/snap/snap_pair_author_rec_iter_5.caffemodel"
modelFile = "network/pairs_deploy.prototxt"

sys.argv = ['', 'data/pairs_train_mean.binaryproto', 'data/pairs_train_mean.npy']
execfile("data/convertMean.py")
meanTest = np.load("data/pairs_train_mean.npy")
net = caffe.Classifier(modelFile, pretrainedFile,
                       mean = meanTest.mean(1).mean(1), #   [scalar for 1 channel]
                       raw_scale=255
                       , image_dims=(96, 200)
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

imS0 = np.swapaxes(im, 1, 0)
imS0 = np.swapaxes(imS0, 1, 2)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['pair_data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', meanTest.mean(1).mean(1)) # mean pixel
transformer.set_input_scale('data', 0.00390625)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


net.blobs['pair_data'].data[...] = transformer.preprocess('data', imS0)
scipy.misc.imsave('inputData0.jpg', net.blobs['pair_data'].data[0][0])
out = net.forward()

outputSlicer = net.blobs['data'].data
outputSlicerP = net.blobs['data_p'].data
scipy.misc.imsave('inputToSlicer.jpg', outputSlicer[0, 0, :, :])

#tmp = net.blobs['pair_data'].data
#inputItest = transformer.deprocess('data', tmp)
#scipy.misc.imsave('inputItest.jpg', inputItest[:, :, 0])

print 'Input to network: shape', imS0.shape, 'mean', imS0.mean(), ', min' ,imS0.min(), ', max', imS0.max()
print 'Ouput slicer: shape', outputSlicer.shape, 'mean', outputSlicer.mean(), ', min' ,outputSlicer.min(), ', max', outputSlicer.max()
print 'Ouput slicer p: shape', outputSlicerP.shape, 'mean', outputSlicerP.mean(), ', min' ,outputSlicerP.min(), ', max', outputSlicer.max()

################ conv
print ''
outConv1 = net.blobs['conv1'].data
outConv1p = net.blobs['conv1_p'].data
print 'Output conv1 Min, max' , outConv1.min(), ' ', outConv1.max()
print 'Output conv1_p Min, max' , outConv1p.min(), ' ', outConv1p.max()

print ''
outConv2 = net.blobs['conv2'].data
outConv2p = net.blobs['conv2_p'].data
print 'Output conv2 Min, max' , outConv2.min(), ' ', outConv2.max()
print 'Output conv2_p Min, max' , outConv2p.min(), ' ', outConv2p.max()


############## fc
print ''
wFc1 = net.params['fc1'][0].data
wFc1p = net.params['fc1_p'][0].data
print 'Weights for fc1, mean', wFc1.mean(), ', min' ,wFc1.min(), ', max', wFc1.max()
print 'Weights for fc1_p, mean', wFc1p.mean(), ', min' ,wFc1p.min(), ', max', wFc1p.max()
bFc1 = net.params['fc1'][1].data
#print wFc1
outFc1 = net.blobs['fc1'].data[0]
outFc1p = net.blobs['fc1_p'].data[0]
print 'Output fc1: mean', outFc1.mean(), ', min' ,outFc1.min(), ', max', outFc1.max()
print 'Output fc1_p: mean', outFc1p.mean(), ', min' ,outFc1p.min(), ', max', outFc1p.max()


print ''
wD = net.params['descriptor'][0].data
wDp = net.params['descriptor_p'][0].data
print 'Weights for descriptor, mean', wD.mean(), ', min' ,wD.min(), ', max', wD.max()
print 'Weights for descriptor_p, mean', wDp.mean(), ', min' ,wDp.min(), ', max', wDp.max()
#print wFc1
outD = net.blobs['descriptor'].data[0]
outDp = net.blobs['descriptor_p'].data[0]
print 'Output descriptor: mean', outD.mean(), ', min' ,outD.min(), ', max', outD.max()
print 'Output descriptor_p: mean', outDp.mean(), ', min' ,outDp.min(), ', max', outDp.max()

#outD = out['descriptor']
#outDp = out['descriptor_p']

