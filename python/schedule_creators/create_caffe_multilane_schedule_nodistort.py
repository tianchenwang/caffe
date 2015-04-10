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
def main(rootdir, batchsize, exclude_list):
    cam_num = 2
    num_distortions = 1
    root_to_prefix = rootdir if rootdir[-1] is not '/' else rootdir[:-1]
    pref = os.path.basename(root_to_prefix) + '_'
    visited_prefix = set([])
    fileList = []
    print rootdir
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: '_interp_lanes.pickle' in z, files)
        files = filter(lambda z: ('tomilpitas_a' not in z) and ('ToMonterey_b' not in z), files)
        #files = filter(lambda z: ('sanrafael_e' in z), files)
        files = filter(lambda z: ('ToMonterey_c' in z), files)
        print files
        for f in files:
          fileList.append(os.path.join(root,f))

    schedule = []
    for f in fileList:
      path, fname = os.path.split(f)
      rootpath, subdir = os.path.split(f)
      rootpath, subdir = os.path.split(rootpath)
      fid = open(f, 'r')
      lanes_data = pickle.load(fid)
      pts_length = int(lanes_data['left'].shape[0]/10)
      fid.close()
      fname = string.replace(fname,'_interp_lanes.pickle', '')
      prefix = path + '/' + fname
      if not os.path.isfile(prefix+'_multilane_points_planar_done.npz'):
        print prefix+'_multilane_points_planar_done.npz not found, skipping...'
        continue
      print prefix+'_multilane_points_planar_done.npz'
      for cam_num in [1,2]:
        if int(subdir[0])<6: # new gps out format and video frame rate starting from june-14
          length = pts_length
        else:
          length = pts_length/2
           
        video_name = path+'/split_0_'+fname+str(cam_num)+'.avi'
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

        length = end_frame-start_frame
        if length<batchsize:
          continue
        num_batches = length/batchsize
        clipped_length = num_batches*batchsize

        for dd in xrange(num_distortions):
          frame_nums = np.arange(start_frame, end_frame)
          persp_ids = np.ones(frame_nums.shape)*6#(frame_nums+dd)%num_distortions # perspective id to use
          #random.shuffle(frame_nums)
          persp_ids = persp_ids[frame_nums-start_frame]
          frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
          persp_ids = persp_ids[0:clipped_length]
          split_frames = np.split(frame_nums, num_batches)
          split_persp = np.split(persp_ids, num_batches)
          for b in xrange(num_batches):
            schedule.append([video_name, split_frames[b], split_persp[b]])
    #random.shuffle(schedule)
    #fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_4-11-14-sanrafael-sanrafael_e1.avi_batch'+str(batchsize)+'.txt', 'w')
    fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_8-1-14-monterey-ToMonterey_c1.avi_batch'+str(batchsize)+'.txt', 'w')
    for s in schedule:
      write_str = s[0]+','+np.array_str(s[1], max_line_width=10000)[1:-1]+','+np.array_str(s[2],max_line_width=10000)[1:-1]+'\n'
      for aa in range(20):
        write_str = string.replace(write_str, '  ', ' ') 
        write_str = string.replace(write_str, ', ', ',') 
      fid.write(write_str)
    fid.close()
    #pickle.dump(schedule, open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule1_batch'+str(batchsize)+'_2cam.pickle', 'wb'))
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

