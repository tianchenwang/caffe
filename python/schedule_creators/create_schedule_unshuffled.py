import os
import sys
import cv, cv2
import commands
import numpy as np
import random
import time
import pickle
"""
Generates a training schedule so that the loader reads directly from videos
"""
def main(rootdir, exclude_list, batchsize):
    num_distortions = 7
    root_to_prefix = rootdir if rootdir[-1] is not '/' else rootdir[:-1]
    pref = os.path.basename(root_to_prefix) + '_'
    visited_prefix = set([])
    fileList = []
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: 'split_0' in z, files)
        files = filter(lambda z: '1.avi' in z, files)
        print root
        for f in files:
            fileList.append(os.path.join(root,f))

    prefix = len(os.path.commonprefix(fileList))
    schedule = []
    for f in fileList:
      path, fname = os.path.split(f)
      fname = fname[8:-5]
      prefix = path + '/' + fname
      if (prefix + '1.avi') in exclude_list:
          print 'skipping: %s' % prefix
          continue
      print f

      capture = cv2.VideoCapture(f)
      length = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
      # get number of frames in the video file.
      # if opencv gives 0 frames, try again using ffmpeg
      if length<10:
        capture.release()
        command = 'ffmpeg -i '+ f+ ' -vcodec copy -f rawvideo -y /dev/null 2>&1 | tr ^M \'\\n\' | awk \'/^frame=/ {print $2}\'|tail -n 1'
        status, lengthout=commands.getstatusoutput(command)
        time.sleep(2)
        status, lengthout=commands.getstatusoutput(command)
        length = int(lengthout)
      if length<96:
        continue
      num_batches = length/batchsize
      clipped_length = num_batches*batchsize
      for dd in xrange(num_distortions):
        frame_nums = np.arange(length)
        persp_ids = (frame_nums+dd)%num_distortions # perspective id to use
        persp_ids = persp_ids[frame_nums]
        frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
        persp_ids = persp_ids[0:clipped_length]
        split_frames = np.split(frame_nums, num_batches)
        split_persp = np.split(persp_ids, num_batches)
        for b in xrange(num_batches):
          schedule.append([f, split_frames[b], split_persp[b]])
      capture.release()
    pickle.dump(schedule, open('/afs/cs.stanford.edu/u/twangcat/scratch/unshuffled_schedule.pickle', 'wb'))
    #print schedule
      



if __name__ == '__main__':
    f = open(sys.argv[2], 'rb')
    lines = f.readlines()
    f.close()
    test = False
    for i in xrange(len(lines)):
        lines[i] = lines[i].strip()
    batch_size = 96
    main(sys.argv[1], lines, batch_size)

