import os
import sys
import cv, cv2
import commands
import numpy as np
import random
import time
import pickle
"""
Generates a testing schedule so that the loader reads directly from videos
"""
def main(rootdir, batchsize):
    fileList = []
    path, basename = os.path.split(rootdir)
    if path == '':
      path = './'
    for j in range(1):
      fileList.append(path + '/' + 'split_' + str(j) + '_' + basename)

    schedule = []
    for f in fileList:
      path, fname = os.path.split(f)
      fname = fname[8:-5]
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
      if length<batchsize+30:
        continue
      length -= 30
      #length=3000
      num_batches = length/batchsize
      clipped_length = num_batches*batchsize
      frame_nums = np.arange(length)
      persp_ids = 6*np.ones(length, dtype=np.int64) # perspective id to use
      persp_ids = persp_ids[frame_nums]
      frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
      persp_ids = persp_ids[0:clipped_length]
      split_frames = np.split(frame_nums, num_batches)
      split_persp = np.split(persp_ids, num_batches)
      for b in xrange(num_batches):
        schedule.append([f, split_frames[b], split_persp[b]])
      capture.release()
    pickle.dump(schedule, open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_test_schedule_4-10-14-pleasanton-'+basename+'_'+str(batchsize)+'.pickle', 'wb'))
    #print schedule
      



if __name__ == '__main__':
    batch_size = int(sys.argv[2])
    main(sys.argv[1], batch_size)

