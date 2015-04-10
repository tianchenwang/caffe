import os
import sys
import commands
import numpy as np
import random
import time
import pickle
import string
import cv,cv2
"""
Generates a test schedule so that the loader reads directly from videos
"""
def main(rootdir, batchsize, exclude_list):
    cam_num = 2
    root_to_prefix = rootdir if rootdir[-1] is not '/' else rootdir[:-1]
    dummy, folder = os.path.split(root_to_prefix)
    fileList = []
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: '_interp_lanes.pickle' in z, files)
        print files 
        if True:#'4-10-14-pleasanton' not in root:
          print root
          for f in files:
              fileList.append(os.path.join(root,f))

    schedule = []
    for f in fileList:
      path, fname = os.path.split(f)
      
      fid = open(f, 'r')
      lanes_data = pickle.load(fid)
      pts_length = int(lanes_data['left'].shape[0]/10)
      fid.close()
      fname = string.replace(fname,'_interp_lanes.pickle', '')
      prefix = path + '/' + fname
      for cam_num in [1]:
        for sp in xrange(10):
          video_name = path+'/split_'+str(sp)+'_'+fname+str(cam_num)+'.avi'
          capture = cv2.VideoCapture(video_name)
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
          #length = pts_length
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

          frame_nums = np.arange(start_frame, end_frame)
          persp_ids = 6*np.ones(length, dtype=np.int64) # perspective id to use                                       
          #persp_ids = 2*np.ones(length, dtype=np.int64) # perspective id to use                                       
          persp_ids = persp_ids[frame_nums-start_frame]
          frame_nums = frame_nums[0:clipped_length] # take batches of 'batchsize' and throw away remainders.
          persp_ids = persp_ids[0:clipped_length]
          split_frames = np.split(frame_nums, num_batches)
          split_persp = np.split(persp_ids, num_batches)
          print video_name+' '+str(num_batches)+' batches'
          for b in xrange(num_batches):
            schedule.append([video_name, split_frames[b], split_persp[b]])
    pickle.dump(schedule, open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_test_schedule_'+folder+ '_batch'+str(batchsize)+'.pickle', 'wb'))
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

