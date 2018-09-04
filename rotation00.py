import numpy as np
import cv2
from glob import glob
import os
from sys import argv

dataset_input = "/data/mydata/darknet3_zoo/data_bottle03"
#dataset_input = "/data/mydata/darknet3_zoo/rotation_lab"
dataset_output = "tmpimg"
dir_image = dataset_input+"/images/"
dir_label = dataset_input+"/labels/"

time_interval = 0
labeled_only  = 0
save_video    = 0

image_names = sorted(glob(dir_image+"/*"))
print("# of images: %d"%len(image_names))

if save_video:
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  height_image, width_image = cv2.imread(image_names[0]).shape[:2]
  output = cv2.VideoWriter('output.avi',fourcc, 50.0, (width_image,height_image))

def yolo_xymm(bbox_yolo,height_image,width_image):
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

def xymm_yolo(label_xymm,height_image,width_image):
  category = label_xymm[0]
  x_left   = label_xymm[1]
  y_top    = label_xymm[2]
  x_right  = label_xymm[3]
  y_bottom = label_xymm[4]
  x_center_bbox = (x_left  +x_right )/2. / width_image
  y_center_bbox = (y_top   +y_bottom)/2. / height_image
  width_bbox    = (x_right -x_left  )*1. / width_image
  height_bbox   = (y_bottom-y_top   )*1. / height_image
  return category, x_center_bbox, y_center_bbox, width_bbox, height_bbox

for image_name in image_names:
  print(image_name)
  label_name = dir_label+"/"+os.path.splitext(os.path.basename(image_name))[0]+".txt"
  with open(label_name) as f:
    labels0 = f.readlines()

  if len(labels0) != 0:
    image0 = cv2.imread(image_name)
    height_image, width_image = image0.shape[:2]
    labels0_xymm = []
    for bbox_yolo in labels0:
      bbox_yolo = bbox_yolo.strip("\n").split()
      labels0_xymm.append( yolo_xymm(bbox_yolo,height_image,width_image) )
  else:
    if labeled_only:
      continue

  for angle in range(0,360,30):
    if not angle:
      image = np.array(image0)
      labels = list(labels0)
    else:
      center = int(width_image/2), int(height_image/2)
      scale = 1.
      matrix = cv2.getRotationMatrix2D(center,angle, scale)
      image = cv2.warpAffine(image0,matrix,(width_image,height_image),borderMode=cv2.BORDER_REPLICATE)
      #image = cv2.warpAffine(image0,matrix,(width_image,height_image),borderMode=cv2.BORDER_REFLECT_101)

    for label0_xymm in labels0_xymm:
      category, x_left, y_top, x_right, y_bottom = label0_xymm
      if angle != 0:
        points0 = np.array([[x_left,y_top,1.], [x_left,y_bottom,1.], [x_right,y_top,1.], [x_right,y_bottom,1.]])
        points = np.dot(matrix,points0.T).T
        x_left   = int(min(map(lambda p: p[0], points)))
        x_right  = int(max(map(lambda p: p[0], points)))
        y_top    = int(min(map(lambda p: p[1], points)))
        y_bottom = int(max(map(lambda p: p[1], points)))
      print([category, x_left, y_top, x_right, y_bottom])
      label_bbox = xymm_yolo([category, x_left, y_top, x_right, y_bottom],height_image,width_image)
      print(label_bbox)
      category, x_left, y_top, x_right, y_bottom = yolo_xymm(label_bbox,height_image,width_image)
      print([category, x_left, y_top, x_right, y_bottom])
      cv2.rectangle(image,(x_left,y_top),(x_right,y_bottom),(255,255,255),2)
 
    cv2.imshow("test",image)
    if save_video:
      output.write(image)
 
    if cv2.waitKey(time_interval) & 0xff == ord('q'):
      quit()
