import cv2
import os
import numpy as np
imgpath="./data_dir/"
outputpath="./data_dir_crop_y/"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
sum = 1
for imgx in os.listdir(imgpath):
  a, b = os.path.splitext(imgx)
  tmp = cv2.imread(imgpath + a +b)
  x,y,z = tmp.shape
  img_size = 100
  coords_x = int(x/img_size)
  coords_y = int(y/img_size)
  for i in range(0,coords_x+1):
    for j in range(0,coords_y+1):
      if((i+1)*img_size > x or (j+1)*img_size > y):
        break
      img = tmp[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size]
      #print(str(i*img_size) + "   "+ str((i+1)*img_size))
      #print(str(j*img_size)+ "   "+ str((j+1)*img_size))
      cv2.imwrite(outputpath + str(sum) + '.jpg', img)
      sum = sum +1 
  print(str(sum)+"ok")