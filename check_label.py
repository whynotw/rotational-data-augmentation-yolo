import numpy as np
import cv2
from glob import glob
import os
from sys import argv
import argparse

#parsing
parser = argparse.ArgumentParser()
parser.add_argument("-i",dest="dataset_input",
                         help="directory containing data you want to visualize.",
                         required=True)
parser.add_argument("-t",dest="time_interval",
                         help="time interval to control speed of displaying images",
                         default=1,
                         type=int)
parser.add_argument("-l",dest="labeled_only",
                         help="only visualizing for data with label",
                         default=0,
                         type=float)
parser.add_argument("-s",dest="save_video",
                         help="save video or not, filename is out.avi",
                         default=0,
                         type=int)

args = parser.parse_args()
dataset_input = args.dataset_input
time_interval = args.time_interval
labeled_only = args.labeled_only
save_video = args.save_video

#if len(argv) == 1:
#  print("$1: dataset name")
#  quit()
#else:
#  dataset_input = argv[1]
#quit()

dir_image = dataset_input+"/images/"
dir_label = dataset_input+"/labels/"

image_names = sorted(glob(dir_image+"/*"))
print("# of images: %d"%len(image_names))

if save_video:
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  height_image, width_image = cv2.imread(image_names[0]).shape[:2]
  output = cv2.VideoWriter('output.avi',fourcc, 50.0, (width_image,height_image))

def yolo_xymm(bbox_yolo,height_image,width_image):
  bbox_yolo = bbox_yolo.strip("\n").split()
  category      =       bbox_yolo[0]
  x_center_bbox = float(bbox_yolo[1])
  y_center_bbox = float(bbox_yolo[2])
  width_bbox    = float(bbox_yolo[3])
  height_bbox   = float(bbox_yolo[4])
  x_left   = int( (x_center_bbox- width_bbox/2.) * width_image )
  x_right  = int( (x_center_bbox+ width_bbox/2.) * width_image )
  y_top    = int( (y_center_bbox-height_bbox/2.) * height_image )
  y_bottom = int( (y_center_bbox+height_bbox/2.) * height_image )
  return category, x_left, y_top, x_right, y_bottom

for image_name in image_names:
  print(image_name)
  label_name = dir_label+"/"+os.path.splitext(os.path.basename(image_name))[0]+".txt"
  with open(label_name) as f:
    labels = f.readlines()

  if len(labels) == 0 and labeled_only:
    continue
  else:
    image = cv2.imread(image_name)
    height_image, width_image = image.shape[:2]

  for bbox_yolo in labels:
    category, x_left, y_top, x_right, y_bottom = yolo_xymm(bbox_yolo,height_image,width_image)
    cv2.rectangle(image,(x_left,y_top),(x_right,y_bottom),(255,255,255),1)

  cv2.imshow("test",image)
  if save_video:
    output.write(image)

  if cv2.waitKey(time_interval) & 0xff == ord('q'):
    quit()
