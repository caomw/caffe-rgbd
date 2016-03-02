import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


imgfolder = '/scratch/16824/data/crop_imgs/'
test_listfile = '/home/xiaolonw/assignment/data/testlist_class.txt'
result_file = 'bbox_results.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/home/xiaolonw/cnncode/caffe-coco/coco_scripts/exp_bbox/test.prototxt',
                '/scratch/16824/test/exp_bbox/model__iter_50000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))

print(batch_count)

f = open(result_file, 'w')
for i in range(batch_count):
	print(i)

	out = net.forward()
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		lbl = int(test_list[id].split(' ')[1])
		fname = test_list[id].split(' ')[0]
		
		prop = out['fc8_bbox'][j]
		fname2 = imgfolder + fname 
		img = caffe.io.load_image(fname2)
		h = np.shape(img)[0]
		w = np.shape(img)[1]
		prop[0] = prop[0] * w
		prop[1] = prop[1] * h
		prop[2] = prop[2] * w
		prop[3] = prop[3] * h 

		f.write(fname)
		f.write(' ')
		for k in range(len(prop)):
			f.write('{0: f}'.format(prop[k]))
		f.write('\n')

f.close()







