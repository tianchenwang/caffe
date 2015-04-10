import os
import sys
import commands
import numpy as np
import random
import time
import pickle
import string
"""
Generates a training schedule so that the video data loader in caffe reads directly from videos
"""
def main(schedule_file, batchsize):
    num_distortions = 7
    schedule = []
    with open(schedule_file, 'r') as sid:
      for line in sid:
        aa = string.split(line,'avi')
        if len(aa)==1:
          continue
        video_name = aa[0]+'avi'
        frames = np.fromstring(aa[1],sep=' ',dtype='i4')
        start_frame = 0
        length = frames.shape[0]
        if length<batchsize:
          continue
        num_batches = length/batchsize
        clipped_length = num_batches*batchsize

        for dd in xrange(num_distortions):
          frame_nums = frames
          frame_ids = np.arange(length)
          persp_ids = (frame_ids+dd)%num_distortions # perspective id to use
          random.shuffle(frame_nums)
          frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
          persp_ids = persp_ids[0:clipped_length]
          split_frames = np.split(frame_nums, num_batches)
          split_persp = np.split(persp_ids, num_batches)
          for b in xrange(num_batches):
            schedule.append([video_name, split_frames[b], split_persp[b]])
    random.shuffle(schedule)
    fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule_Vshape_batch'+str(batchsize)+'_strict2_2cam.txt', 'w')
    for s in schedule:
      if 'to_granada_e' not in s[0]:
        write_str = s[0]+','+np.array_str(s[1], max_line_width=10000)[1:-1]+','+np.array_str(s[2],max_line_width=10000)[1:-1]+'\n'
        for aa in range(20):
          write_str = string.replace(write_str, '  ', ' ') 
          write_str = string.replace(write_str, ', ', ',') 
        fid.write(write_str)
    fid.close()
    #pickle.dump(schedule, open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule1_batch'+str(batchsize)+'_2cam.pickle', 'wb'))
    print len(schedule)
      



if __name__ == '__main__':
    batch_size = int(sys.argv[2])
    main(sys.argv[1], batch_size)

