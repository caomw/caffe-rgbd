from __future__ import division
import sys

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('/nfs.yoda/xiaolonw/fast_rcnn/models/pre_gan_joint4/fast_rcnn_zero.caffemodel')


solver.step(50000)


