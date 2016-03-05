import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/nfs/hn38/users/xiaolonw/sunrgbd/SUNRGBDtoolbox/testlist2.txt'
result_file = 'cls_results2.txt'

caffe.set_device(1)
caffe.set_mode_gpu()
                #'/scratch/16824/test/exp_cls/model__iter_50000.caffemodel',
net = caffe.Net('/nfs/hn46/xiaolonw/render_cnncode/caffe-rgbd/render_scripts/pre_cls_1fc_nobnft/test.prototxt',
                '/nfs.yoda/xiaolonw/fast_rcnn/models_sunrgbd/pre_cls_1fc_nobnft/model__iter_20000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
accuracy = 0

f = open(result_file, 'w')
print(batch_count)
for i in range(batch_count):

	out = net.forward()
	print(i)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		lbl = int(test_list[id].split(' ')[2])
		fname = test_list[id].split(' ')[0]
		
		prop = out['cls_prob'][j] 
		pred_lbl = prop.argmax()
		if pred_lbl == lbl:
			accuracy = accuracy + 1

		f.write(fname)
		f.write('{0: d}'.format(pred_lbl))
		f.write('\n')

f.close()

accuracy = accuracy * 1.0 / ( data_counts) 

print accuracy


