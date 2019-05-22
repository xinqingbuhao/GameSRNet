from model import EDSR
import scipy.misc
import cv2
import argparse
import data
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=32,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=10000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
imgpath="./data/"
starttime = datetime.datetime.now()
sum=0
for imgx in os.listdir(imgpath):
  a, b = os.path.splitext(imgx)
  x = scipy.misc.imread(imgpath+a+".jpg")
  #x=cv2.medianBlur(x,3)
  x=cv2.bilateralFilter(x,3,100,100)
  scipy.misc.imsave(args.outdir+"/"+a+"f.jpg",x)
  inputs = x
  outputs = network.predict(x)
  #scipy.misc.imsave(args.outdir+"/input_"+a+".jpg",inputs)
  #outputs=scipy.misc.imresize(outputs,50, interp='bilinear', mode=None)
  scipy.misc.imsave(args.outdir+"/"+a+".jpg",outputs)
  print a+".jpg"
  sum=sum+1
endtime = datetime.datetime.now()
print(starttime)
print(endtime)
print (float((endtime - starttime).seconds)/float(sum))
