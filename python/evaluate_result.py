import numpy as np
import scipy
import matplotlib.pyplot as plt
import caffe
import sys
import Image
import time
import cv2
import argparse

from driving_utils import *

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gt_label', required=True)
  parser.add_argument('--dump_images', action='store_true')
  parser.add_argument('--output_path')
  args = parser.parse_args()

  if args.dump_images:
    assert args.output_path is not None

  net = caffe.Classifier('/deep/u/willsong/caffe/models/brody/deploy.prototxt',
                         '/deep/u/willsong/caffe/models/brody/caffe_brody_train_iter_200000.caffemodel')
  net.set_phase_test()
  net.set_mode_gpu()
  net.set_mean('data', np.load('/deep/u/willsong/caffe/python/driving_mean.npy'))  # ImageNet mean
  net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
  net.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

  tp = 0
  fp = 0
  fn = 0
  for line in open(args.gt_label).readlines():
    tokens = line.split()
    fname = tokens[0]
    bbs = tokens[2:]
    gt_bbs = get_gt_bbs(bbs)

    img_name = fname.split('/')[-1]
    # print img_name, '...',
    start = time.time()
    scores = net.ff([caffe.io.load_image(fname)])
    # print 'done ff, took %f seconds' % (time.time() - start)

    mask = get_mask(net.blobs['pixel-prob'].data[4])
    rects = get_rects(net.blobs['bb-output'].data[4], mask)

    if args.dump_images:
      assert output_path != ''
      image = net.deprocess('data', net.blobs['data'].data[4])
      zoomed_mask = np.empty((480, 640))
      zoomed_mask = scipy.ndimage.zoom(mask, 8, order=0)
      masked_image = image.transpose((2, 0, 1))
      masked_image[0, :, :] += zoomed_mask
      masked_image = np.clip(masked_image, 0, 1)
      masked_image = masked_image.transpose((1, 2, 0))
      boxed_image = np.copy(masked_image)
      if len(rects) > 0:
        boxed_image = draw_rects(boxed_image, rects)
      Image.fromarray(
          (boxed_image * 255).astype('uint8')).save(args.output_path + '/' + img_name)

    used_rect = set()
    for bb in gt_bbs:
      matched = False
      for i, rect in enumerate(rects):
        if i in used_rect:
          continue
        if bb.jaccard(rect) > 0.5:
          tp += 1
          used_rect.add(i)
          matched = True
          break
      if not matched:
        fn += 1
    fp += len(rects) - len(used_rect)
    if tp + fp > 0 and tp + fn > 0:
      print 'Precision: %f,  Recall: %f' % (float(tp) / (tp + fp), float(tp) / (tp + fn))

if __name__ == '__main__':
  main()
