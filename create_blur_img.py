import cv2
import os
import numpy as np
imgpath="./data_dir_crop_y/"
outputpath="./data_dir_crop_x/"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
sum =1
for imgx in os.listdir(imgpath):
  a, b = os.path.splitext(imgx)
  img = cv2.imread(imgpath + a +b)
  img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
  #img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(outputpath + a + b, img)
  sum =sum +1
print(str(sum)+"ok")