
# from https://github.com/BVLC/caffe/issues/290

import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
    print "Usage: python convertMean.py pairs_test_mean.binaryproto pairs_test_mean.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )