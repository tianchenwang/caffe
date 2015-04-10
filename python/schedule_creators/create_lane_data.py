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
from numpy import array, dot, zeros, around, divide, ones
from scipy.io import loadmat
import string
from transformations import euler_matrix
from CameraParams import getCameraParams
from GPSReprojection import PointsMask
from CameraReprojection import pixelTo3d
from WGS84toENU import deg2rad

"""
Generates a training schedule so that the loader reads directly from videos
"""


def main(rootdir, laneLoc):

    lane_data = dict()
    cam = getCameraParams()
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
            #pitch = -cam_to_use['rot_x']
            #height = 1.106
            #R_camera_pitch = euler_matrix(cam_to_use['rot_x'], cam_to_use['rot_y'], cam_to_use['rot_z'], 'sxyz')[0:3, 0:3]
            Tc = np.eye(4)
            #Tc[0:3, 0:3] = R_camera_pitch.transpose()
            #Tc[0:3, 3] = [-0.2, -height, -0.5]
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
