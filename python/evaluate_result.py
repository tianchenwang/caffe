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

  #net = caffe.Classifier('/deep/u/willsong/caffe/models/brody/deploy.prototxt',
  #                       '/deep/u/willsong/caffe/models/brody/caffe_brody_train_iter_200000.caffemodel')
  net = caffe.Classifier('/deep/u/willsong/caffe/models/brody/deploy_deeppy.prototxt',
                         '/scail/group/deeplearning/sail-deep-gpu/brodyh/pynet/l7_maps96,256,384,384,384,4096,4096_rf11,5,3,3,3,6,1,1_lcnl0,1_img640,480_do1_reg5e-4_pst2,2,0,0,2,0,0_mom0.90_bbms32_md8_mbs8_bbr10_L1_dpth0.05_wr_su_srk0.75_mo0.75_bcs8_jimg672_doc5,6_2.netparameter')
  net.set_phase_test()
  net.set_mode_gpu()
  net.set_mean('data', np.load('/deep/u/willsong/caffe/python/driving_mean_r.npy'))  # ImageNet mean
  net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
  #net.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

  tp = 0
  fp = 0
  fn = 0
  for line in open(args.gt_label).readlines():
    tokens = line.split()
    fname = tokens[0]
    bbs = tokens[2:]
    gt_bbs = get_gt_bbs(bbs)

    # filter out small boxes
    filtered_gt_bbs = []
    for bb in gt_bbs:
      if bb.xmax - bb.xmin < 8:
        continue
      filtered_gt_bbs.append(bb)
    gt_bbs = filtered_gt_bbs

    # print img_name, '...',
    start = time.time()
    scores = net.ff([caffe.io.load_image(fname)])
    # print 'done ff, took %f seconds' % (time.time() - start)

    mask, display_mask = get_mask(net.blobs['pixel-prob'].data[4])
    rects = get_rects(net.blobs['bb-output-tiled'].data[4][:4,:,:], mask, 16)

    used_rect = set()
    unmatched_gt = set()
    for j, bb in enumerate(gt_bbs):
      matched = False
      for i, rect in enumerate(rects):
        if i in used_rect:
          continue
        if bb.jaccard(rect) > 0.3:
          tp += 1
          used_rect.add(i)
          matched = True
          break
      if not matched:
        fn += 1
        unmatched_gt.add(j)
    fp += len(rects) - len(used_rect)
    if tp + fp > 0 and tp + fn > 0:
      print 'Precision: %f,  Recall: %f' % (float(tp) / (tp + fp), float(tp) / (tp + fn))

    unmatched_gts = [bb for i, bb in enumerate(gt_bbs) if i in unmatched_gt]
    matched_rects = [bb for i, bb in enumerate(rects) if i in used_rect]
    unmatched_rects = [bb for i, bb in enumerate(rects) if i not in used_rect]
    if args.dump_images:
      img_name = fname.split('/')[-1]
      dump_image(net, display_mask, matched_rects, args.output_path + img_name, \
          unmatched_gts, unmatched_rects)

if __name__ == '__main__':
  main()
