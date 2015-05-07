from numpy import array, dot, zeros, around, divide, ones
import cv,cv2
from transformations import euler_matrix
import numpy as np
import logging
import pickle
import string
import os
import time
from GPSReader import GPSReader
from GPSTransforms import *
from GPSReprojection import *
from WarpUtils import warpPoints
from Q50_config import *
from ArgParser import *
from SetPerspDist import setPersp
__all__=['MultilaneLabelReader']


blue = np.array([255,0,0])
green = np.array([0,255,0])
red = np.array([0,0,255])

def dist2color(dist, max_dist = 90.0):
  # given a distance and a maximum distance, gives a color code for the distance.
  # red being closest, green is mid-range, blue being furthest
  alpha = (dist/max_dist)
  if alpha<0.5:
    color = red*(1-alpha*2)+green*alpha*2
  else:
    beta = alpha-0.5
    color = green*(1-beta*2)+blue*beta*2
  return color.astype(np.int)

def colorful_line(img, start, end, start_color, end_color, thickness):
  # similar to cv.line, but draws a line with gradually (linearly) changing color. 
  # allows starting and ending color to be specified. 
  # implemented using recursion.
  if ((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5<=thickness*2:
    cv2.line(img, start, end ,start_color,thickness)
    return img
  mid = (int((start[0]+end[0])/2),int((start[1]+end[1])/2))
  mid_color = [int((start_color[0]+end_color[0])/2),int((start_color[1]+end_color[1])/2),int((start_color[2]+end_color[2]))/2]
  img = colorful_line(img, start, mid, start_color, mid_color, thickness)
  img = colorful_line(img, mid, end, mid_color, end_color, thickness)
  return img



class MultilaneLabelReader():
    def __init__(self, rank, predict_depth = True, batchSize = 48, imdepth=3, imwidth=640, imheight=480, markingWidth=0.07, distortion_file='/scail/group/deeplearning/driving_data/perspective_transforms.pickle', pixShift=0, label_dim = [80,60], new_distort=False, readVideo=False):
      self.outname = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule_Vshape_strict2.txt'
      self.new_distort = new_distort
      if new_distort:
        self.Ps = setPersp()
      else:
        self.Ps = pickle.load(open(distortion_file, 'rb'))
      self.rank = rank
      self.lane_values = dict()
      self.gps_data1 = dict()
      self.gps_times1 = dict()
      self.gps_times2 = dict()
      if os.path.isfile('/deep/group/driving_data/twangcat/caffe_results/laneVisible_cache.pickle'):
        lane_fid = open('/deep/group/driving_data/twangcat/caffe_results/laneVisible_cache.pickle', 'rb')
        self.laneVisible = pickle.load(lane_fid)
      else:
        self.laneVisible = dict()
      self.tr1 = dict()
      self.markingWidth = markingWidth
      self.pixShift = pixShift
      self.labelw = label_dim[0]
      self.labelh = label_dim[1]
      self.labeld = 6 if predict_depth else 4
      self.imwidth = imwidth
      self.imheight = imheight
      self.imdepth = imdepth
      self.label_scale = None
      self.img_scale = None
      self.predict_depth = predict_depth
      self.griddim=4
      self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,255),(128,255,128),(255,128,128),(128,128,0),(128,0,128),(0,128,128),(0,128,255),(0,255,128),(128,0,255),(128,255,0),(255,0,128),(255,128,0)]
      # arrays to adjust for rf offset
      self.x_adj = (np.floor(np.arange(label_dim[0])/self.griddim)*self.griddim+self.griddim/2)*imwidth/label_dim[0]
      self.y_adj = (np.floor(np.arange(label_dim[1])/self.griddim)*self.griddim+self.griddim/2)*imheight/label_dim[1]
      self.y_adj = np.array([self.y_adj]).transpose()
      #self.weight_labels= np.ones([batchSize, self.labelh, self.labelw,1],dtype='f4',order='C')
      self.time1 = 0
      self.time2 = 0
      self.time3 = 0
      self.time4 = 0
      self.time5 = 0
      self.timer_cnt = 1
      self.iter = 0
      self.vid_names = set() 

    def runBatch(self, vid_name, gps_filename1, gps_dat, gps_times1, gps_times2, frames, lanes, tr1,Pid, split_num, cam_num, params):
        if vid_name not in self.vid_names:
          #fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule_Vshape_raw.txt','a')
          fid = open(self.outname,'a')
          self.vid_names.add(vid_name)
          fid.write('\n'+vid_name)
          fid.close()
        cam = params['cam'][cam_num-1]#self.cam[cam_num - 1]
        lidar_height = params['lidar']['height']
        T_from_l_to_i = params['lidar']['T_from_l_to_i']
        T_from_i_to_l = np.linalg.inv(T_from_l_to_i) 
        starting_point = 4#12
        meters_per_point = 86#24#12#6
        points_fwd = 2#6#12
        starting_point2 = 15#12
        points_fwd2 = 15#6#12
        scan_range = starting_point + (points_fwd-1)*meters_per_point
        seconds_ahead=2
        output_num = 0
        batchSize = frames.shape[0] 
        count = 0
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0
        computed_lanes = 0
        if gps_filename1 not in self.laneVisible:
          num_frames = int((gps_times2.shape[0]+1-split_num)/10)
          self.laneVisible[gps_filename1]=np.ones([num_frames,lanes['num_lanes']], dtype='bool')
        for idx in xrange(batchSize):
            frame = frames[idx]
            fnum2 =frame*10+split_num-1 # global video frame. if split0, *10+9; if split1, *10+0; if split 2, *10+1 .... if split9, *10+8
            #fnum2 =frame # global video frame. if split0, *10+9; if split1, *10+0; if split 2, *10+1 .... if split9, *10+8
            if cam_num>2:
              fnum2 *=2 # wideview cams have half the framerate
            t = gps_times2[fnum2] # time stamp for the current video frame (same as gps_mark2)
            fnum1 = Idfromt(gps_times1,t) # corresponding frame in gps_mark1
            if self.new_distort:
              T = self.Ps['T'][Pid[idx]]
              P=self.Ps['M_'+str(cam_num)][Pid[idx]]
            else:
              T = np.eye(4)
              P = self.Ps[Pid[idx]]
            # car trajectory in current camera frame
            local_pts = MapPos(tr1[fnum1:fnum1+290,0:3,3], tr1[fnum1,:,:], cam, T_from_i_to_l)
            local_pts[1,:]+=lidar_height # subtract height to get point on ground
            # transform according to real-world distortion
            local_pts = np.vstack((local_pts, np.ones((1,local_pts.shape[1]))))
            local_pts = np.dot(T, local_pts)[0:3,:]
            # pick start and end point frame ids
            ids = np.where(np.logical_and(gps_times1>t-seconds_ahead*1000000, gps_times1<t+seconds_ahead*1000000))[0]
            ids = range(ids[0], ids[-1]+1)
            for l1 in np.where(self.laneVisible[gps_filename1][frame,:])[0]:
              for l2 in np.where(self.laneVisible[gps_filename1][frame,:])[0]:
                if l2<l1:
                  lane_key1 = 'lane'+str(l1)
                  lane1 = lanes[lane_key1]
                  lane_key2 = 'lane'+str(l2)
                  lane2 = lanes[lane_key2]
                  ahead1 = np.dot(tr1[fnum1,0:3,0:3], (lane1[0,:]-tr1[fnum1,0:3,3]))[0] # num of meters the starting of a lane is in front of car.
                  side1 = np.dot(tr1[fnum1,0:3,0:3], (lane1[0,:]-tr1[fnum1,0:3,3]))[1] # num of meters the starting of a lane is laterall away from car.
                  ahead2 = np.dot(tr1[fnum1,0:3,0:3], (lane2[0,:]-tr1[fnum1,0:3,3]))[0] # num of meters the starting of a lane is in front of car.
                  side2 = np.dot(tr1[fnum1,0:3,0:3], (lane2[0,:]-tr1[fnum1,0:3,3]))[1] # num of meters the starting of a lane laterally away from car.
                  if (ahead1<40 and ahead1>10 and side1<6 and side1>-6) or (ahead2<40 and ahead2>10 and side2<6 and side2>-6):
                    if np.min(np.sum((lane2 - lane1[0:1,:])**2,axis=1))<0.5 or np.min(np.sum((lane1 - lane2[0:1,:])**2,axis=1))<0.5:
                      fid = open(self.outname,'a')
                      fid.write(' '+str(frame))
                      fid.close()



    def runLabelling(self, f, frames, Pid): # filename, frame numbers, transformation ids

        time1=0
        Pid = Pid.tolist()
        cam_num = int(f[-5])
        splitidx = string.index(f,'split_')
        split_num = int(f[splitidx+6])
        if split_num==0:
          split_num=10
        path, fname = os.path.split(f)
        fname = fname[8:] # remove 'split_?'
        args = parse_args(path, fname)
        prefix = path + '/' + fname

        params = args['params'] 
        cam = params['cam'][cam_num-1]
        self.label_scale = np.array([[float(self.labelw) / cam['width']], [float(self.labelh) / cam['height']]])
        self.img_scale = np.array([[float(self.imwidth) / cam['width']], [float(self.imheight) / cam['height']]])
        if os.path.isfile(args['gps_mark2']):
          gps_key1='gps_mark1'
          gps_key2='gps_mark2'
          postfix_len = 13
        else:
          gps_key1='gps'
          gps_key2='gps'
          postfix_len=8
        
        # gps_mark2 
        gps_filename2= args[gps_key2]
        time0 = time.time()
    
        if not (gps_filename2 in self.gps_times2): # if haven't read this gps file before, cache it in dict.
          gps_reader2 = GPSReader(gps_filename2)
          gps_data2 = gps_reader2.getNumericData()
          self.gps_times2[gps_filename2] = utc_from_gps_log_all(gps_data2)
        gps_times2 = self.gps_times2[gps_filename2]
        # gps_mark1
        gps_filename1= args[gps_key1]
        if not (gps_filename1 in self.gps_times1): # if haven't read this gps file before, cache it in dict.
          gps_reader1 = GPSReader(gps_filename1)
          self.gps_data1[gps_filename1] = gps_reader1.getNumericData()
          self.tr1[gps_filename1]=IMUTransforms(self.gps_data1[gps_filename1])
          self.gps_times1[gps_filename1] = utc_from_gps_log_all(self.gps_data1[gps_filename1])
        gps_data1 = self.gps_data1[gps_filename1]
        tr1 = self.tr1[gps_filename1]
        gps_times1 =self.gps_times1[gps_filename1]
        time1 += time.time()-time0

        prefix = gps_filename2[0:-postfix_len]
        lane_filename = prefix+'_multilane_points_planar_done.npz'
        if not (lane_filename in self.lane_values):
          self.lane_values[lane_filename] = np.load(lane_filename)
        lanes = self.lane_values[lane_filename] # these values are alread pre-computed and saved, now just read it from dictionary
        self.time1+=time1
        print str(self.iter)+' '+f#+ ' '+ str(frames)
        self.runBatch(f, gps_filename1, gps_data1, gps_times1, gps_times2, frames, lanes, tr1, Pid, split_num, cam_num, params)
        self.iter+=1


if __name__ == '__main__':
  label_reader = MultilaneLabelReader( 0,label_dim = [80,60], predict_depth=True)
  schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_train_schedule1_batch40_2cam_nodistort.txt'
  with open(schedule_file, 'r') as sid:
    for line in sid:                                                                                            
      words = string.split(line, ',')                                                                           
      batch = []                                                                                                
      batch.append(words[0])                                                                                    
      batch.append(np.fromstring(words[1], dtype=int, sep=' '))                                                 
      batch.append(np.fromstring(words[2], dtype=int, sep=' '))
      label_reader.runLabelling(batch[0], batch[1], batch[2])
