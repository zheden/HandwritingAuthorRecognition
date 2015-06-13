
import caffe
import lmdb
import numpy as np
import scipy

# get image from testing dataset
env = lmdb.open("data/pairs_test_lmdb", readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000026') # diff authors here

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flatIm = np.fromstring(datum.data, dtype=np.uint8)
im = flatIm.reshape(datum.channels, datum.height, datum.width)
scipy.misc.imsave('testIm1.jpg', im[0, :, :])
scipy.misc.imsave('testIm2.jpg', im[1, :, :])


caffe.set_mode_cpu()
pretrainedFile = "network/snap/snap_pair_author_rec_iter_200.caffemodel"
modelFile = "network/pairs_deploy.prototxt"

meanTest = np.load("data/pairs_test_mean.npy")
net = caffe.Classifier(modelFile, pretrainedFile,
                       mean=meanTest.mean(0), #   h x w ???
                       raw_scale=255,
                       image_dims=(171, 379)
                       )

im1 = im[0, :, :]
im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
im2 = im[1, :, :]
im2 = im2.reshape(im2.shape[0], im2.shape[1], 1)
descriptor1 = net.predict([im1])
descriptor2 = net.predict([im2])
print descriptor1
print descriptor2