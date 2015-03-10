import numpy as np
import pickle
__all__=['PerspectiveReader']

class PerspectiveReader():
    def __init__(self, distortion_file='/scail/group/deeplearning/driving_data/perspective_transforms.pickle'):
      self.Ps = pickle.load(open(distortion_file, 'rb'))
    def read(self):
      return self.Ps
