
import caffe
import lmdb
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
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
    scipy.misc.imsave('filters.jpg', data[:, :, 0])
    return data

# get image from testing dataset
env = lmdb.open("data/pairs_train_lmdb", readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000026') # diff authors here

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flatIm = np.fromstring(datum.data, dtype=np.uint8)
im = flatIm.reshape(datum.channels, datum.height, datum.width)
scipy.misc.imsave('testIm1.jpg', im[0, :, :])
scipy.misc.imsave('testIm2.jpg', im[1, :, :])


caffe.set_mode_cpu()
pretrainedFile = "network/snap1/snap_pair_author_rec_iter_60000.caffemodel"
modelFile = "network/pairs_deploy.prototxt"

sys.argv = ['', 'data/pairs_test_mean.binaryproto', 'data/pairs_test_mean.npy']
execfile("data/convertMean.py")
meanTest = np.load("data/pairs_test_mean.npy")
net = caffe.Classifier(modelFile, pretrainedFile,
                       mean=meanTest.mean(0), #   h x w ???
                       raw_scale=255,
                       image_dims=(96, 300)
                       )

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
ff = filters.transpose(0, 2, 3, 1)
print ff.shape
dd = vis_square(ff)

print 'weights for fc1:'
print net.blobs['fc1'].data[0]

print 'descriptor:'
im1 = im[0, :, :]
im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
descriptor1 = net.predict([im1])
print descriptor1