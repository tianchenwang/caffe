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


def main(rootdir):

    gps_data = dict()
    for root, subfolders, files in os.walk(rootdir):
        files = filter(lambda z: '_gps.out' in z, files)
        for f in files:
            path=root
            fname=f[0:-8]
            prefix = path + '/' + fname
            gps_filename = prefix +  '_gps.out'
              
            frames_to_skip = 0  #int(sys.argv[3]) if len(sys.argv) > 3 else 0
            final_frame = -1  #int(sys.argv[4]) if len(sys.argv) > 4 else -1
            prefix = gps_filename[0:-8]
            gps_reader = GPSReader(gps_filename)
            gps_dat = gps_reader.getNumericData()

            gps_data[gps_filename]=gps_dat
    print 'writing to disk...'
    pickle.dump(gps_data, open('/scail/group/deeplearning/driving_data/twangcat/gps_data.pickle','wb'))


if __name__ == '__main__':
    main(sys.argv[1])
