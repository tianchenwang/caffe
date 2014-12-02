import numpy as np
import scipy
import matplotlib.pyplot as plt
import caffe
import sys
import Image
import time
import cv2
import argparse
import pickle

from driving_utils import *

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gt_label', required=True)
  parser.add_argument('--output_prefix', required=True)
  parser.add_argument('--model_prefix', required=True)
  parser.add_argument('--deploy', required=True)
  parser.add_argument('--iter', required=True)
  parser.add_argument('--dump_images', action='store_true')
  parser.add_argument('--image_output_path')
  parser.add_argument('--interval', type=int, default=10000)
  parser.add_argument('--sampling_increment', type=int, default=1)
  args = parser.parse_args()

  if args.dump_images:
    assert args.image_output_path is not None

  if '-' not in args.iter:
    iters = [args.iter]
  else:
    begin, end = args.iter.split('-')
    iters = [str(i) for i in range(int(begin), int(end), args.interval)]

  print 'Generating results for iterations', ' '.join(iters)

  for iter in iters:
    print '#### Begin to generate detection results for iteration', iter
    net = caffe.Classifier(args.deploy, args.model_prefix + '_iter_' + iter + '.caffemodel')
#    net = caffe.Classifier('/deep/u/willsong/caffe/models/brody/deploy.prototxt',
#                           '/deep/u/willsong/caffe/models/brody/caffe_brody_train_iter_200000.caffemodel')
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data', np.load('/deep/u/willsong/caffe/python/driving_mean.npy'))  # ImageNet mean
    net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    all_rects = []
    if args.dump_images:
      image_folder_name = args.image_output_path + '_' + iter + '/'
      if not os.path.exists(image_folder_name):
        os.makedirs(image_folder_name)
    inc = 0
    for i, line in enumerate(open(args.gt_label).readlines()):
      inc += 1
      if inc < args.sampling_increment:
        continue
      else:
        inc = 0
      tokens = line.split()
      fname = tokens[0]

      # print img_name, '...',
      # start = time.time()
      scores = net.ff([caffe.io.load_image(fname)])
      # print 'done ff, took %f seconds' % (time.time() - start)

      mask = get_mask(net.blobs['pixel-prob'].data[4])
      rects = get_rects(net.blobs['bb-output'].data[4], mask)
      all_rects.append(rects)

      if args.dump_images:
        img_name = fname.split('/')[-1]
        dump_image(net, mask, rects, image_folder_name + img_name)

    pickle.dump(all_rects, open(args.output_prefix + '_' + iter + '.pkl', 'w'))

if __name__ == '__main__':
  main()
