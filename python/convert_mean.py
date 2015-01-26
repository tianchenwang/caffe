"""
Conver mean binary proto to npy so that it can be used for visualizing
network activity.
"""
import os
import numpy as np
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2
from PIL import Image

def main(argv):
  if len(argv) != 2:
    print 'Usage: %s mean_binary' % os.path.basename(sys.argv[0])
    return

  mean_data = caffe_pb2.BlobProto()
  mean_data.ParseFromString(open(sys.argv[1]).read())

  mean_data = np.array(mean_data.data)
  print mean_data.shape
  mean_img = mean_data.reshape([3, 480, 640])
#  mean_img = mean_img[(2, 1, 0), :, :]
#  np.save(open('new_driving_mean.npy', 'wb'), mean_img)

  mean_img = np.transpose(mean_img, (1, 2, 0))
  Image.fromarray(mean_img.astype('uint8')).save('test_mean.png')

  """
  real_img = caffe.io.load_image( \
      '/deep/group/driving_data/andriluka/IMAGES/driving_data_q50_data/all_extracted/4-2-14-monterey-split_0_280S_a2/4-2-14-monterey-split_0_280S_a2_000341.jpeg')
  real_img = caffe.io.resize_image(real_img * 255, (480, 640, 3))
  Image.fromarray(real_img.astype('uint8')).save('original.png')
  Image.fromarray(np.clip(real_img - mean_img, 0, 255).astype('uint8')).save('sub.png')
  """


if __name__ == '__main__':
  import sys
  main(sys.argv)
