import os
import sys
import cv, cv2
import commands
import numpy as np
import random
import time
import pickle
from GPSReader import GPSReader
from GPSTransforms import GPSTransforms
from GPSReprojection import *
from WGS84toENU import *
from numpy import array, dot, zeros, around, divide, ones
from scipy.io import loadmat
import string
from transformations import euler_matrix
from CameraParams import getCameraParams
"""
Generates a training schedule so that the loader reads directly from videos
"""
def pixelTo3d(pixels, cam):
  pitch = -cam['rot_x'] # constants based on camera setup
  height = 1.105 # constants based on camera setup
  N = pixels.shape[0]
  assert(pixels.shape[1] == 2)
  Z = ((pixels[:,1]-cam['cv'])*sin(pitch)*height+cam['fy']*cos(pitch)*height)/(cos(pitch)*(pixels[:,1]-cam['cv'])-cam['fy']*sin(pitch))
  X = (cos(pitch)*Z-sin(pitch)*height)*(pixels[:,0]-cam['cu'])/cam['fx']
  Y = np.zeros((N))
  return np.vstack((X,Y,Z)).transpose()

def PointsMask(pos_wrt_camera, Camera):
    vpix = np.around(np.dot(Camera['KK'], np.divide(pos_wrt_camera, pos_wrt_camera[2,:])))
    vpix = vpix.astype(np.int32)
    return vpix

def warpPoints(P, pts):
    """
warpPoints takes a list of points and performs a matrix transform on them

P is a perspective transform matrix (3x3)
pts is a 2xN matrix of (x, y) coordinates to be transformed
"""
    pts = np.vstack([pts, np.ones((1, pts.shape[1]))])
    out = np.dot(P, pts)
    return out[0:2] / out[2]

def deg2rad(angleInDeg):
  return (3.1415926536 / 180) * angleInDeg
'''
def getCameraParams():
cam = [{}, {}]
for i in [0, 1]:
cam[i]['R_to_c_from_i'] = np.array([[-1, 0, 0],
[0, 0, -1],
[0, -1, 0]])

if i == 0:
cam[i]['rot_x'] = deg2rad(-0.8) # better cam 1
cam[i]['rot_y'] = deg2rad(-0.5)
cam[i]['rot_z'] = deg2rad(-0.005)
cam[i]['t_x'] = -0.5
cam[i]['t_y'] = 1.1
cam[i]['t_z'] = 0.0
elif i == 1:
cam[i]['rot_x'] = deg2rad(-0.61) # better cam 2
cam[i]['rot_y'] = deg2rad(0.2)
cam[i]['rot_z'] = deg2rad(0.0)
cam[i]['t_x'] = 0.5
cam[i]['t_y'] = 1.1
cam[i]['t_z'] = 0.0

cam[i]['fx'] = 2221.8
cam[i]['fy'] = 2233.7
cam[i]['cu'] = 623.7
cam[i]['cv'] = 445.7
cam[i]['KK'] = np.array([[cam[i]['fx'], 0.0, cam[i]['cu']],
[0.0, cam[i]['fy'], cam[i]['cv']],
[0.0, 0.0, 1.0]])
cam[i]['f'] = (cam[i]['fx'] + cam[i]['fy']) / 2

return cam
'''
def main(rootdir, laneLoc):

    lane_data = dict()
    cam = getCameraParams()
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: '_gps.out' in z, files)
        for f in files:
            path=root
            fname=f[0:-8]
            prefix = path + '/' + fname
            gps_filename = prefix + '_gps.out'
              
            frames_to_skip = 0 #int(sys.argv[3]) if len(sys.argv) > 3 else 0
            final_frame = -1 #int(sys.argv[4]) if len(sys.argv) > 4 else -1
            prefix = gps_filename[0:-8]
            gps_reader = GPSReader(gps_filename)
            gps_dat = gps_reader.getNumericData()
            cam_to_use = cam[0]


            path, output_base = os.path.split(prefix)
            path2 = string.replace(path,'320x240_2', 'twangcat/raw_data_backup')
            lanefile_prefix = string.replace(path2, '/','_')
            pts_file = laneLoc + lanefile_prefix+'-' + output_base + str(1)+ '-interpolated.mat'
            key = (lanefile_prefix+'-'+output_base + str(1))
            if key in lane_data:
              print key + ' already exist, skipping'
              continue
            if not os.path.isfile(pts_file):
              print 'failed to read pts_file from '+pts_file+', skipping'
              continue
            print 'processing '+pts_file
            pts = loadmat(pts_file)
            #pts = loadmat(laneLoc+'_scail_group_deeplearning_driving_data_twangcat_raw_data_backup_8-15-tracy-gilroy_east-580E_c1-interpolated.mat')
            lp = pixelTo3d(pts['left'], cam_to_use)
            rp = pixelTo3d(pts['right'], cam_to_use)
            tr = GPSTransforms(gps_dat, cam_to_use)
            pitch = -cam_to_use['rot_x']
            height = 1.106
            R_camera_pitch = euler_matrix(cam_to_use['rot_x'], cam_to_use['rot_y'], cam_to_use['rot_z'], 'sxyz')[0:3, 0:3]
            Tc = np.eye(4)
            Tc[0:3, 0:3] = R_camera_pitch.transpose()
            Tc[0:3, 3] = [-0.2, -height, -0.5]
            lpts = np.zeros((lp.shape[0], 4))
            rpts = np.zeros((rp.shape[0], 4))
            for t in range(min(tr.shape[0], lp.shape[0])):
                lpts[t, :] = np.dot(tr[t, :, :], np.linalg.solve(Tc, np.array([lp[t, 0], lp[t, 1], lp[t, 2], 1])))
                rpts[t, :] = np.dot(tr[t, :, :], np.linalg.solve(Tc, np.array([rp[t, 0], rp[t, 1], rp[t, 2], 1])))
            ldist = np.apply_along_axis(np.linalg.norm, 1, np.concatenate((np.array([[0, 0, 0, 0]]), lpts[1:] - lpts[0:-1])))
            rdist = np.apply_along_axis(np.linalg.norm, 1, np.concatenate((np.array([[0, 0, 0, 0]]), rpts[1:] - rpts[0:-1])))
            lane_data[key]=[lpts, rpts, ldist, rdist, tr]
    print 'writing to disk...'
    pickle.dump(lane_data, open('/scail/group/deeplearning/driving_data/twangcat/lanes_data.pickle','wb'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
