import cv,cv2

vid_name = '/scail/group/deeplearning/driving_data/q50_data/7-25-14-monterey/split_0_ToMonterey_c3.avi'
cap = cv2.VideoCapture(vid_name)

for i in range(10000):
  cap.set(cv.CV_CAP_PROP_POS_FRAMES, i)
  success, img = cap.read()
  if not success:
    break
  print img.shape
  x1 = 1020-190
  x2 = x1+380
  y1 = 560 - 142
  y2 = y1+285
  img2 = cv2.resize(img[y1:y2, x1:x2,:], (640,480))
  cv2.imwrite('/scr/twangcat/caffenet_results/test/cropped/'+str(i)+'.png', img2)
