from __future__ import division
import sys

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(2)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
# solver.net.copy_from('/scratch/16824/models/bvlc_reference_caffenet.caffemodel')


solver.step(100000)


