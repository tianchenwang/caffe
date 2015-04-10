import os
import sys
import commands
import numpy as np
import random
import time
import pickle
import string
"""
Generates a training schedule so that the loader reads directly from videos
"""
def main(rootdir, batchsize, exclude_list):
    hard_list=pickle.load(open('/scail/group/deeplearning/driving_data/twangcat/schedules/hard_frames.pickle','r'))
    cam_num = 2
    num_distortions = 7
    root_to_prefix = rootdir if rootdir[-1] is not '/' else rootdir[:-1]
    pref = os.path.basename(root_to_prefix) + '_'
    visited_prefix = set([])
    fileList = []
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: '_interp_lanes.pickle' in z, files)
        rootpath, subdir = os.path.split(root)
        if len(subdir)>0 and int(subdir[0])<6 and '4-20-14' not in root and '4-29-14' not in root and '4-30-14' not in root and '5-1-14' not in root and '5-7-14' not in root:
          print subdir
          for f in files:
              fileList.append(os.path.join(root,f))

    schedule = []
    for f in fileList:
      print f
      path, fname = os.path.split(f)
      fid = open(f, 'r')
      lanes_data = pickle.load(fid)
      pts_length = int(lanes_data['left'].shape[0]/10)
      fid.close()
      fname = string.replace(fname,'_interp_lanes.pickle', '')
      prefix = path + '/' + fname
      hard_key = string.replace(prefix+'2.avi', '640x480_Q50', 'q50_data')
      for cam_num in [1,2]:
        if hard_key in hard_list.keys():
          hard_frames = hard_list[hard_key]
          for ss in range(10):
            length = pts_length
            video_name = path+'/split_'+str(ss)+'_'+fname+str(cam_num)+'.avi'
            print video_name
            start_frame = 0
            end_frame = length-30
            if (prefix) in exclude_list:
              if exclude_list[prefix][0]>-0.5 and exclude_list[prefix][1]>-0.5:
                start_frame = int(exclude_list[prefix][0]*length)
                if exclude_list[prefix][1]==1:
                  end_frame = length-30
                else:
                  end_frame = int(exclude_list[prefix][1]*length)
                print 'file: %s, %d:%d' %( prefix, start_frame, end_frame)
              elif exclude_list[prefix][0]==-1 and exclude_list[prefix][1]==-1:
                print 'skipping: %s' % prefix
                continue
              else:
                print 'wrong exclude list format!'
                assert 0==1
            print prefix

            frame_nums = np.arange(start_frame, end_frame)
            frame_nums = np.intersect1d(frame_nums, hard_frames)
            length = frame_nums.shape[0]
            if length<batchsize:
              continue
            num_batches = length/batchsize
            clipped_length = num_batches*batchsize

            for dd in xrange(num_distortions):
              persp_ids = (frame_nums+dd)%num_distortions # perspective id to use
              random.shuffle(frame_nums)
              persp_ids = persp_ids[0:frame_nums.shape[0]]
              frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
              persp_ids = persp_ids[0:clipped_length]
              split_frames = np.split(frame_nums, num_batches)
              split_persp = np.split(persp_ids, num_batches)
              for b in xrange(num_batches):
                schedule.append([video_name, split_frames[b], split_persp[b]])
    random.shuffle(schedule)
    pickle.dump(schedule, open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_train_schedule_batch_hard'+str(batchsize)+'_2cam.pickle', 'wb'))
    print len(schedule)
      



if __name__ == '__main__':
    f = open(sys.argv[2], 'rb')
    lines = f.readlines()
    f.close()
    exclude_files = dict() # list of files to (partially or fully) exclude
    for i in xrange(len(lines)):
      aa = lines[i].strip().split()
      print len(aa)
      if len(aa)==3: # specified starting and ending fractions
        assert float(aa[1])>=0 and float(aa[2])>float(aa[1]) and float(aa[2])<=1
        exclude_files[aa[0]] = [float(aa[1]), float(aa[2])]
      elif len(aa)==1: # didn't specify starting and ending, skip the entire video
        exclude_files[aa[0]] = [-1, -1]
      else:
        assert 0==1
        
    batch_size = int(sys.argv[3])
    main(sys.argv[1], batch_size, exclude_files)

