import lmdb
import glob
import argparse
import fnmatch
import os
import sys
import cv2

import numpy as np
import cPickle as pickle
import caffe.proto.caffe_pb2 as cpb

def main():
    args = get_args()
    define_paths(args)
    bb_files = find(args.bounding_box_dir, '*verified.pkl')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        os.chmod(args.output, 0755)
    env = lmdb.Environment(args.output,
                           map_size=1024**4, # 1TB
                           mode=0664)
    keys = []
    data_idx = 0
    with env.begin(write=True) as txn:
        for bb_file_idx, bb_file in enumerate(bb_files):
            run_str = '%s   %d/%d' % ('/'.join(bb_file.split('/')[-3:-1]),
                                    bb_file_idx,
                                    len(bb_files))
            bbs_batch = pickle.load(open(bb_file))
            image_files = bbfile2images(args, bb_file)
            assert len(bbs_batch) == len(image_files)
            for idx, (image_file, bbs) in \
                enumerate(zip(image_files, bbs_batch)):
                if idx % 15 == 0:
                    sys.stdout.write("\r%s   %0.2f%%   total images: %d" %
                                     (run_str,
                                      float(idx)*100/len(image_files),
                                     data_idx))
                    sys.stdout.flush()
                key, value = get_key_value(args, image_file, bbs)
                txn.put(key, value)
                keys.append(key)
                data_idx += 1
            sys.stdout.write('\n')

        # write DataBaseInfo as first key
        dbi = cpb.DataBaseInfo()
        dbi.keys.extend(sorted(keys + [chr(0)]))
        txn.put(chr(0), dbi.SerializeToString())


def get_key_value(args, image_file, bbs):
    data = cpb.DrivingData()
    datum = data.car_image_datum
    old_size = read_image_to_datum(image_file, args.width,
                                   args.height, datum)
    factor = [args.width, args.height] / np.array(old_size, dtype='f4')
    factor = np.tile(factor, 2)
    data.car_img_source = image_file
    for bb in bbs:
        car_box = data.car_boxes.add()
        rect = [int(r) for r in (bb['rect'] * factor).round()]
        car_box.xmin = rect[0]
        car_box.ymin = rect[1]
        car_box.xmax = rect[0] + rect[2]
        car_box.ymax = rect[1] + rect[3]
        car_box.img_width = args.width
        car_box.img_height = args.height

    key = imgname2keyname(args, image_file)
    value = data.SerializeToString()
    return key, value


def imgname2keyname(args, fname):
    p = os.path
    key = fname[len(args.image_dir):]
    while key[0] == '/':
        key = key[1:]
    frame_ind = int(p.splitext(p.basename(key))[0])
    key = '%s/%08d' % (p.dirname(key), frame_ind)
    return key


def read_image_to_datum(fname, width, height, datum):
    ext = os.path.splitext(fname)[1]
    img = cv2.imread(fname)
    orig_size = img.shape[:-1][::-1]
    img = cv2.resize(img, (width, height))
    assert img.shape[2] == 3
    success, imgcode = cv2.imencode(ext, img)
    assert success
    datum.ClearField('data')
    datum.ClearField('float_data')
    datum.data = ''.join([chr(i) for i in imgcode])
    datum.encoded = True
    datum.width = img.shape[1]
    datum.height = img.shape[0]
    datum.channels = img.shape[2]
    return orig_size


def bbfile2images(args, bb_file):
    driving_run_dir = os.path.dirname(bb_file[len(args.bounding_box_dir):])
    image_files = glob.glob(args.image_dir + '/'
                            + driving_run_dir
                            + args.camera_postfix + '/*')
    return image_files


def find(root_dir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
      for filename in fnmatch.filter(filenames, pattern):
          matches.append(os.path.join(root, filename))
    return matches


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-o', '--output', dest='output', type=str,
                      help='Output folder for data')
    parser.add_argument('-w', '--width', dest='width', type=int,
                      help='output image width')
    parser.add_argument('-h', '--height', dest='height', type=int,
                      help='output image height')
    args = parser.parse_args()
    return args


def define_paths(args):
    args.bounding_box_dir = '/deep/group/driving/bounding-boxes'
    args.image_dir = '/deep/group/driving_data/q50_data'
    args.camera_postfix = '601'


if __name__ == '__main__':
    main()
