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
    def __init__(self, rank, depth_only = False, label_width=80, label_height=60, griddim=4, batchSize = 48, predict_depth = True, imdepth=3, imwidth=640, imheight=480, markingWidth=0.07, distortion_file='/scail/group/deeplearning/driving_data/perspective_transforms.pickle', pixShift=0, new_distort=False, readVideo=False):
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
      self.labelw = label_width
      self.labelh = label_height
      if depth_only:
        self.labeld = 1
      else:
        self.labeld = 6 if predict_depth else 4
      self.imwidth = imwidth
      self.imheight = imheight
      self.imdepth = imdepth
      self.label_scale = None
      self.img_scale = None
      self.predict_depth = predict_depth
      self.depth_only = depth_only
      self.griddim=griddim
      self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,255),(128,255,128),(255,128,128),(128,128,0),(128,0,128),(0,128,128),(0,128,255),(0,255,128),(128,0,255),(128,255,0),(255,0,128),(255,128,0)]
      # arrays to adjust for rf offset
      self.x_adj = (np.floor(np.arange(label_width)/self.griddim)*self.griddim+self.griddim/2)*imwidth/label_width
      self.y_adj = (np.floor(np.arange(label_height)/self.griddim)*self.griddim+self.griddim/2)*imheight/label_height
      self.y_adj = np.array([self.y_adj]).transpose()
      #self.weight_labels= np.ones([batchSize, self.labelh, self.labelw,1],dtype='f4',order='C')
      self.time1 = 0
      self.time2 = 0
      self.time3 = 0
      self.time4 = 0
      self.time5 = 0
      self.timer_cnt = 1
      self.iter = 0 
      '''
      lane_filename = '/scail/group/deeplearning/driving_data/640x480_Q50/4-3-14-gilroy/from_gilroy_c_multilane_points_planar_done'+str(self.rank)+'.pickle'
      lfid = open(lane_filename,'rb')
      self.lanes = pickle.load(lfid)
      lfid.close()
      #self.lanes = np.load(lane_filename)
      '''

          
        

    def runBatch(self, vid_name, gps_filename1, gps_dat, gps_times1, gps_times2, frames, lanes, tr1,Pid, split_num, cam_num, params):
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
        seconds_ahead=5
        output_num = 0
        batchSize = frames.shape[0] 
        self.labels= np.zeros([batchSize,self.labelh, self.labelw,1],dtype='f4',order='C')
        self.reg_labels= np.zeros([batchSize, self.labelh, self.labelw,self.labeld],dtype='f4',order='C')
        count = 0
        #fid = open('/deep/group/driving_data/twangcat/caffe_models/pylog_rank'+str(self.rank), 'a')
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
            #temp_label = np.zeros([self.labelh, self.labelw], dtype='f4',order='C')
            #if (not self.depth_only) and self.predict_depth:
            #  temp_reg1 = np.zeros([self.labelh, self.labelw, self.labeld/2],dtype='f4',order='C')
            #  temp_reg2 = np.zeros([self.labelh, self.labelw, self.labeld/2],dtype='f4',order='C')
            #else:
            #  temp_reg = np.zeros([self.labelh, self.labelw, self.labeld],dtype='f4',order='C')
            #temp_weight = np.ones([self.labelh, self.labelw, 1],dtype='f4',order='C')
            #for l in range(lanes['num_lanes']):
            for l in np.where(self.laneVisible[gps_filename1][frame,:])[0]:
              time0 = time.time()
              #time.sleep(0.0002)
              time.sleep(0.01)
              #if not self.laneVisible[gps_filename1][frame, l]:
              #  # already know this lane is not visible at this frame. just skip.
              #  continue
              computed_lanes+=1
              lane_key = 'lane'+str(l)
              lane = lanes[lane_key]
              # find the appropriate portion on the lane (close to the position of car, in front of camera, etc)
              # find the closest point on the lane to the two end-points on the trajectory of car. ideally this should be done before-hand to increase efficiency.
              dist_near = np.sum((lane-tr1[ids[0],0:3,3])**2, axis=1) # find distances of lane to current 'near' position.
              dist_far = np.sum((lane-tr1[ids[-1],0:3,3])**2, axis=1) # find distances of lane to current 'far' position.
              dist_self = np.sum((lane-tr1[fnum1,0:3,3])**2, axis=1) # find distances of lane to current self position.
              dist_mask = np.where(dist_self<=(scan_range**2))[0]# only consider points to be valid within scan_range from the 'near' position
              if len(dist_mask)==0:
                self.laneVisible[gps_filename1][frame, l]=False
                continue
              time2 += time.time()-time0
              time0 = time.time()
              nearid = np.argmin(dist_near[dist_mask]) # for those valid points, find the one closet to 'near' position.
              farid = np.argmin(dist_far[dist_mask])  #and far position
              lids = range(dist_mask[nearid], dist_mask[farid]+1) # convert back to global id and make it into a consecutive list.
              lane3d = MapPos(lane[lids,:], tr1[fnum1,:,:], cam,T_from_i_to_l) # lane markings in current camera frame
              if np.all(lane3d[2,:]<=0):
                continue
              lane3d = lane3d[:,lane3d[2,:]>0] # make sure in front of camera
              dense_border=40
              # make points a nx denser for the closer points (<40 meters), so lanes in label mask look contiguous.
              for aaa in range(5):
                nearidx = lane3d[2,:]<dense_border
                if np.sum(nearidx)>1 and np.abs(lane3d[0,0])<10: # at least 2 pts within 20 meters, ego-lane only
                  borderidx = np.where(nearidx)[0][-1] # the idx closest to the 20 meter mark
                  nearidx = np.arange(borderidx+1)
                  nearpts = lane3d[:,nearidx]
                  newpts = (nearpts[:,1:]+nearpts[:,:-1])/2 # add a bunch of mid pts
                  new_lane3d = np.zeros([3, lane3d.shape[1]+borderidx])
                  new_lane3d[:, 0:(borderidx*2+1):2] = nearpts
                  new_lane3d[:,1:(borderidx*2):2] = newpts
                  # now the rest of the pts further than 20 meters
                  new_lane3d[:,(borderidx*2+1):] = lane3d[:,borderidx+1:]
                  lane3d = new_lane3d
              depths = lane3d[2,:]
              # project into 2d image
              (c, J)  = cv2.projectPoints(lane3d[0:3,:].transpose(), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
              # need to define lane id. If necessary split current lane based on lateral distance. 
              c= warpPoints(P, c[:,0,:].transpose()[0:2,:])
              # scale down to the size of the label mask 
              labelpix = np.transpose(np.round(c*self.label_scale))
              # scale down to the size of the actual image 
              imgpix = c*self.img_scale
              labelpix = labelpix[::-1,:] 
              imgpix = imgpix[:,::-1]
              depths = depths[::-1]
              # find unique indices to be marked in the label mask
              lu = np.ascontiguousarray(labelpix).view(np.dtype((np.void, labelpix.dtype.itemsize * labelpix.shape[1])))
              _, l_idx = np.unique(lu, return_index=True)
              l_idx = np.sort(l_idx) 
              labelpix = (np.transpose(labelpix[l_idx,:])).astype('i4')
              imgpix = imgpix[:,l_idx]
              depths = depths[l_idx]
              # draw labels on temp masks
              mask_color=1
              # remove labels that are out of bound
              good_idx_x = np.logical_and(labelpix[0,:]>-1, labelpix[0,:]<self.labelw)
              good_idx_y = np.logical_and(labelpix[1,:]>-1, labelpix[1,:]<self.labelh)
              good_idx = np.logical_and(good_idx_x, good_idx_y)
              labelpix = labelpix[:,good_idx]
              imgpix = imgpix[:,good_idx]
              depths = depths[good_idx]
              time3 += time.time()-time0
              labelpix = labelpix[:,::-1] 
              imgpix = imgpix[:,::-1]
              depths = depths[::-1]
              if self.depth_only:
                self.reg_labels[idx, labelpix[1,:], labelpix[0,:], 0] = depths
              else:
                self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 0] = imgpix[0,0:-3] # x1
                self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 1] = imgpix[1,0:-3] # y1
                self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 2] = imgpix[0,2:-1] # x2
                self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 3] = imgpix[1,2:-1] # y2
                if self.predict_depth:
                  self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 4] = depths[0:-3] # depth1
                  self.reg_labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 5] = depths[2:-1] # depth2
              self.labels[idx, labelpix[1,1:-2], labelpix[0,1:-2], 0] = mask_color 
              '''
              for ii in range(1,imgpix.shape[1]-1):
                time00 = time.time()
                ip = ii-1
                ic = ii
                xp = labelpix[0,ip]
                yp = labelpix[1,ip]
                xc = labelpix[0,ic]
                yc = labelpix[1,ic]
               
                x1 = xp
                y1 = yp
                x2 = xc
                y2 = yc 
                time4+=time.time()-time00
                time00 = time.time()
                #if yc>-1 and yc<self.labelh and xc>-1 and xc<self.labelw:# and np.abs(yp-yc)<5:
                # only update info for the first pt if nothing has been drawn for this grid. otherwise keep the first point and update the second point.
                if temp_label[yc,xc]<1:
                  regx1 = imgpix[0,ip]
                  regy1 = imgpix[1,ip]
                  depth1 = depths[ip]
                else:
                  if self.predict_depth:
                    regx1 = float(temp_reg1[yc,xc,0])
                    regy1 = float(temp_reg1[yc,xc,1])
                    depth1 = float(temp_reg2[yc,xc,1])
                  else:
                    regx1 = float(temp_reg[yc,xc,0])
                    regy1 = float(temp_reg[yc,xc,1])
                regx2 = imgpix[0,ii+1]
                regy2 = imgpix[1,ii+1]
                depth2 = depths[ii+1]
                if self.predict_depth:
                  if np.abs(x2-x1)<2 and np.abs(y2-y1)<2:
                    temp_reg1[y1,x1] = [regx1, regy1, regx2]
                    temp_reg2[y1,x1] = [regy2, depth1, depth2]
                    temp_reg1[y2,x2] = [regx1, regy1, regx2]
                    temp_reg2[y2,x2] = [regy2, depth1, depth2]
                  else:
                    cv2.line(temp_reg1, (x1,y1), (x2,y2) , [regx1, regy1, regx2], thickness=1 )
                    cv2.line(temp_reg2, (x1,y1), (x2,y2), [regy2, depth1, depth2], thickness=1 )
                else:
                  cv2.line(temp_reg, (x1,y1), (x2,y2) , [regx1,regy1,regx2,regy2], thickness=1 )
                # draw mask label
                if np.abs(x2-x1)<2 and np.abs(y2-y1)<2:
                  temp_label[y1,x1] = mask_color
                  temp_label[y2,x2] = mask_color
                else:
                  cv2.line(temp_label, (x1, y1), (x2, y2), mask_color, thickness=1 )
                time5+=time.time()-time00
              '''
            '''
            # fill temp masks into actual batch labels
            self.labels[idx,:,:,0] = temp_label
            if self.predict_depth:
              self.reg_labels[idx,:,:,0:3] = temp_reg1
              self.reg_labels[idx,:,:,3:] = temp_reg2    
            else:
              self.reg_labels[idx,:,:,:] = temp_reg    
            #self.weight_labels[idx,:,:,:] = temp_weight
            '''
            if not self.depth_only:
              self.reg_labels[idx,:,:,0]-=self.x_adj
              self.reg_labels[idx,:,:,2]-=self.x_adj
              self.reg_labels[idx,:,:,1]-=self.y_adj
              self.reg_labels[idx,:,:,3]-=self.y_adj



        # reshape a batch of label into the right format.
        # caffe does not support 'output block' sizes >1, so flatten it into the z dimension.
        #print ' herere5'
        label_view = np.transpose(self.labels, [0,3,1,2]).reshape(batchSize, 1, self.labelh//self.griddim, self.griddim, self.labelw//self.griddim, self.griddim, order='C')
        label_grid = np.transpose(label_view,[0,1,3,5,2,4]).reshape(batchSize, self.griddim*self.griddim,self.labelh//self.griddim,self.labelw//self.griddim, order='C')
        reg_view = np.transpose(self.reg_labels, [0,3,1,2]).reshape(batchSize, self.labeld, self.labelh//self.griddim, self.griddim, self.labelw//self.griddim, self.griddim)
        reg_grid = np.transpose(reg_view,[0,1,3,5,2,4]).reshape(batchSize, self.labeld*self.griddim*self.griddim,self.labelh//self.griddim,self.labelw//self.griddim)
        full_label = np.empty([batchSize, (self.griddim**2)*(self.labeld+1), self.labelh//self.griddim, self.labelw//self.griddim], dtype='f4',order='C')
        full_label[:,0:(self.griddim**2),:,:] = label_grid # binary mask label comes first along the z dimension
        self.num_pos = np.sum(label_grid, axis=(1,2,3))
        full_label[:,(self.griddim**2):,:,:] = reg_grid  # followed by the regression labels for each channel.
        #print 'time2: '+str(time2)
        self.time2+=time2
        self.time3+=time3
        self.time4+=time4
        self.time5+=time5
        interval = 30
        if False:#self.timer_cnt%interval==0:
          fid = open('caffe_python_time_rank_cont'+str(self.rank), 'a')
          fid.write('%f %f %f %f %f\n'%(self.time1/interval, self.time2/interval, self.time3/interval, self.time4/interval, self.time5/interval))
          fid.close()
          self.time1 = 0
          self.time2 = 0
          self.time3 = 0
          self.time4 = 0
          self.time5 = 0
        self.timer_cnt+=1
        #fid.close()
        return full_label



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
        #lane_filename = prefix+'_multilane_points_done.npz'
        #lane_filename = '/scail/group/deeplearning/driving_data/640x480_Q50/4-3-14-gilroy/from_gilroy_c_multilane_points_planar_done'+str(self.rank)+'.npz'
        if not (lane_filename in self.lane_values):
          self.lane_values[lane_filename] = np.load(lane_filename)
        lanes = self.lane_values[lane_filename] # these values are alread pre-computed and saved, now just read it from dictionary
        #lanes = np.load(lane_filename)
        #lfid = open(lane_filename,'rb')
        #lanes = pickle.load(lfid)
        #lfid.close()
        #print 'time1: '+str(time1)
        self.time1+=time1
        if self.iter==7000 and (not os.path.isfile('/deep/group/driving_data/twangcat/caffe_results/laneVisible_cache.pickle')):
          if self.rank==0:
            lane_fid = open('/deep/group/driving_data/twangcat/caffe_results/laneVisible_cache.pickle', 'wb')
            pickle.dump(self.laneVisible, lane_fid)
         
        full_label = self.runBatch(f, gps_filename1, gps_data1, gps_times1, gps_times2, frames, lanes, tr1, Pid, split_num, cam_num, params)
        #lanes.close()
        self.iter+=1
        return full_label


if __name__ == '__main__':
  label_reader = MultilaneLabelReader(400, 0, 3, 640, 480, label_dim = [80,60], readVideo = True,predict_depth=True)
  fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_test_schedule_4-2-14-monterey-17S_a2.avi_96.pickle')
  batches = pickle.load(fid)
  fid.close()
  label_reader.start()
  for b in batches:
    label_reader.push_batch(b)

