import numpy as np
import cv2
from glob import glob
import os
from sys import argv
import argparse

# parsing
parser = argparse.ArgumentParser()
parser.add_argument("-i",dest="dataset_input",
                         help="directory containing data you want to rotate.",
                         required=True)
parser.add_argument("-o",dest="dataset_output",
                         help="directory to store generated data. this directory will be made automatically.",
                         default="data_rotational")
parser.add_argument("-t",dest="time_interval",
                         help="time interval to control speed of displaying images.",
                         default=1,
                         type=int)
parser.add_argument("-r",dest="ratio",
                         help="ratio for ignoring bounding box near the edges of image.",
                         default=0.8,
                         type=float)
parser.add_argument("-a",dest="angle_interval",
                         help="angle interval for rotating.",
                         default=30,
                         type=int)
parser.add_argument("-s",dest="save_image",
                         help="save data or not, set to 0 for purely visualizing rotated images.",
                         default=1,
                         type=int)
args = parser.parse_args()
dataset_input = args.dataset_input
dataset_output = args.dataset_output
time_interval = args.time_interval
ratio = args.ratio
angle_interval = args.angle_interval
save_image = args.save_image

dir_input_image = dataset_input+"/images/"
dir_input_label = dataset_input+"/labels/"
dir_output_image = dataset_output+"/images/"
dir_output_label = dataset_output+"/labels/"

if save_image:
  os.system("mkdir -p "+dataset_output)
  os.system("mkdir -p "+dataset_output+"/images")
  os.system("mkdir -p "+dataset_output+"/labels")

image_names = sorted(glob(dir_input_image+"/*"))
print("# of images: %d"%len(image_names))

# label and coord are bounding box of yolo and opencv format respectively
# yolo format: <category> <x center> <y center> <width_bbox> <height_bbox> ; range: 0-1
# opencv format: <category> <x_left> <y_top> <x_right> <y_bottom> ; range: 0-width_image or 0-height_image
def label2coord(label,height_image,width_image):
  category      =       label[0]
  x_center_bbox = float(label[1])
  y_center_bbox = float(label[2])
  width_bbox    = float(label[3])
  height_bbox   = float(label[4])
  x_left   = int( (x_center_bbox- width_bbox/2.) * width_image )
  x_right  = int( (x_center_bbox+ width_bbox/2.) * width_image )
  y_top    = int( (y_center_bbox-height_bbox/2.) * height_image )
  y_bottom = int( (y_center_bbox+height_bbox/2.) * height_image )
  return category, x_left, y_top, x_right, y_bottom

def coord2label(coord,height_image,width_image):
  category =       coord[0]
  x_left   = float(coord[1])
  y_top    = float(coord[2])
  x_right  = float(coord[3])
  y_bottom = float(coord[4])
  x_center_bbox = (x_left  +x_right )/2. / width_image
  y_center_bbox = (y_top   +y_bottom)/2. / height_image
  width_bbox    = (x_right -x_left  )    / width_image
  height_bbox   = (y_bottom-y_top   )    / height_image
  return category, x_center_bbox, y_center_bbox, width_bbox, height_bbox

def constraint(value, lower, upper):
  return max(min(value,upper),lower)

for image_name0 in image_names:
  print(image_name0)
  label_name = dir_input_label+"/"+os.path.splitext(os.path.basename(image_name0))[0]+".txt"
  with open(label_name) as f0:
    labels0 = f0.readlines()

  image0 = cv2.imread(image_name0)
  height_image, width_image = image0.shape[:2]
  coords = []
  for label in labels0:
    label = label.strip("\n").split()
    coord = label2coord(label,height_image,width_image)
    h2 = 2*height_image
    w2 = 2*width_image
    coords.append([coord[0],   coord[1],   coord[2],   coord[3],   coord[4]])
    coords.append([coord[0],  -coord[3],   coord[2],  -coord[1],   coord[4]])
    coords.append([coord[0],w2-coord[3],   coord[2],w2-coord[1],   coord[4]])
    coords.append([coord[0],   coord[1],  -coord[4],   coord[3],  -coord[2]])
    coords.append([coord[0],   coord[1],h2-coord[4],   coord[3],h2-coord[2]])

  for angle in range(0,360,angle_interval):
    image_name = dir_output_image+"/"+os.path.splitext(os.path.basename(image_name0))[0]+"_%03d"%angle+".jpg"
    label_name = dir_output_label+"/"+os.path.splitext(os.path.basename(image_name0))[0]+"_%03d"%angle+".txt"
    print(image_name)
    if angle == 0.:
      image = np.array(image0)
    else:
      center = int(width_image/2), int(height_image/2)
      scale = 1.
      matrix = cv2.getRotationMatrix2D(center,angle, scale)
      #image = cv2.warpAffine(image0,matrix,(width_image,height_image),borderMode=cv2.BORDER_REPLICATE)
      image = cv2.warpAffine(image0,matrix,(width_image,height_image),borderMode=cv2.BORDER_REFLECT_101)
    image_annotated = np.array(image)

    # clean label file
    if save_image:
      with open(label_name,"w") as f1:
        pass

    for coord in coords:
      category, x_left, y_top, x_right, y_bottom = coord
      area0 = (x_right-x_left)*(y_bottom-y_top)
      if angle != 0:
        points0 = np.array([[x_left,y_top,1.], [x_left,y_bottom,1.], [x_right,y_top,1.], [x_right,y_bottom,1.]])
        points = np.dot(matrix,points0.T).T
        x_left   = int(min(map(lambda p: p[0], points)))
        x_right  = int(max(map(lambda p: p[0], points)))
        y_top    = int(min(map(lambda p: p[1], points)))
        y_bottom = int(max(map(lambda p: p[1], points)))
      x_left   = constraint(x_left  ,0,width_image)
      x_right  = constraint(x_right ,0,width_image)
      y_top    = constraint(y_top   ,0,height_image)
      y_bottom = constraint(y_bottom,0,height_image)
      area = (x_right-x_left)*(y_bottom-y_top)
      if area > area0*ratio:
        cv2.rectangle(image_annotated,(x_left,y_top),(x_right,y_bottom),(255,255,255),2) # white bbox
        if save_image:
          with open(label_name,"a") as f1:
            label = coord2label([category, x_left, y_top, x_right, y_bottom],height_image,width_image)
            f1.write(" ".join( map(str,label) )+"\n")
      else:
        if not save_image:
          cv2.rectangle(image_annotated,(x_left,y_top),(x_right,y_bottom),(0,0,255),2) # red bbox

    if save_image:
      cv2.imwrite(image_name,image)
    else:
      cv2.imshow("test",image_annotated)

    if cv2.waitKey(time_interval) & 0xff == ord('q'):
      quit()
