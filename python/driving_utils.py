import numpy as np
import scipy
import cv2

class Rect():
  def __init__(self, xmin, ymin, xmax, ymax):
    if xmax < xmin:
      xmax = xmin
    if ymax < ymin:
      ymax = ymin
    self.xmin = int(xmin)
    self.ymin = int(ymin)
    self.xmax = int(xmax)
    self.ymax = int(ymax)
    self.w = self.xmax - self.xmin
    self.h = self.ymax - self.ymin

  def area(self):
    return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

  def jaccard(self, other):
    xmin = max(self.xmin, other.xmin)
    xmax = min(self.xmax, other.xmax)
    ymin = max(self.ymin, other.ymin)
    ymax = min(self.ymax, other.ymax)
    if ymax >= ymin and xmax >= xmin:
      intersect = (xmax - xmin + 1) * (ymax - ymin + 1)
    else:
      return 0
    return float(intersect) / (self.area() + other.area() - intersect)

  def __repr__(self):
    return '(%d,%d,%d,%d)' % (self.xmin, self.ymin, self.xmax, self.ymax)

  def __str__(self):
    return '(%d,%d,%d,%d)' % (self.xmin, self.ymin, self.xmax, self.ymax)

def get_gt_bbs(bbs):
  assert len(bbs) % 4 == 0
  rbbs = []
  for i in range(0, len(bbs), 4):
    rbbs.append(Rect(*bbs[i:i+4]))
  return rbbs

def draw_rects(image, rects):
  for r in rects:
    image[r.ymin:r.ymax+1, r.xmin:r.xmin+2, 1] = 1
    image[r.ymin:r.ymax+1, r.xmax:r.xmax+2, 1] = 1
    image[r.ymin:r.ymin+2, r.xmin:r.xmax+1, 1] = 1
    image[r.ymax:r.ymax+2, r.xmin:r.xmax+1, 1] = 1
  return image

def get_mask(feat):
  mask = np.empty((60, 80))
  for y in range(15):
    for x in range(20):
      mask[y*4:(y+1)*4, x*4:(x+1)*4] = feat[:, y, x].reshape((4, 4))
  return mask

def dump_image(net, mask, rects, path):
  image = net.deprocess('data', net.blobs['data'].data[4])
  zoomed_mask = np.empty((480, 640))
  zoomed_mask = scipy.ndimage.zoom(mask, 8, order=0)
  masked_image = image.transpose((2, 0, 1))
  masked_image[0, :, :] += zoomed_mask
  masked_image = np.clip(masked_image, 0, 1)
  masked_image = masked_image.transpose((1, 2, 0))
  boxed_image = np.copy(masked_image)
  if len(rects) > 0:
    boxed_image = draw_rects(boxed_image, rects)
  Image.fromarray(
      (boxed_image * 255).astype('uint8')).save(path)


def get_rects(feat, mask):
  hard_mask = np.round(mask + 0.3)
  bb = np.empty((4, 60, 80))
  for y in range(15):
    for x in range(20):
      for c in range(4):
        bb[c, y*4:(y+1)*4, x*4:(x+1)*4] = feat[c*16:(c+1)*16, y, x].reshape((4, 4))

  for c in range(4):
    bb[c, :, :] *= hard_mask

  y_offset = np.array([np.arange(16, 480, 32)]).T
  y_offset = np.tile(y_offset, (1, 20))
  x_offset = np.arange(16, 640, 32)
  x_offset = np.tile(x_offset, (15, 1))
  y_offset = scipy.ndimage.zoom(y_offset, 4, order=0)
  x_offset = scipy.ndimage.zoom(x_offset, 4, order=0)
  bb[0, :, :] += x_offset
  bb[2, :, :] += x_offset
  bb[1, :, :] += y_offset
  bb[3, :, :] += y_offset

  selected_rects = hard_mask > 0
  num_rects = np.sum(selected_rects)
  rects = np.empty((num_rects, 4))
  for i in range(4):
    rects[:, i] = bb[i, selected_rects]
  rects = rects[np.logical_and((rects[:, 2] - rects[:, 0]) > 0, (rects[:, 3] - rects[:, 1]) > 0), :]
  rects[:, (2, 3)] -= rects[:, (0, 1)]
  rects = np.clip(rects, 0, 640)
  rects = [rects[i, :] for i in range(rects.shape[0])]
  rects, scores = cv2.groupRectangles(rects, 4, 0.4)

  rectangles = []
  if len(rects) == 0:
    return rectangles
  for i in range(rects.shape[0]):
    rectangles.append(Rect(rects[i, 0], rects[i, 1], rects[i, 0] + rects[i, 2], rects[i, 1] + rects[i, 3]))
  return rectangles
