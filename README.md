# Rotational Data Augmentation for YOLO

## Requirement
	numpy
	opencv-python

## Before and after

Before: the block at left-top corner is bounded by a white box.
![image] https://github.com/whynotw/rotational-data-augmentation-yolo/blob/master/before.png

After: Image is rotated by 30 degree and the block is still bounded by the white box.
![image] https://github.com/whynotw/rotational-data-augmentation-yolo/blob/master/after.png

## Directory structure
	.
	├── check_label.py
	├── rotation.py
	├── data_original
	│   ├── images
	│   │   └── test00.png
	│   └── labels
	│       └── test00.txt
	└── data_rotational
	    ├── images
	    │   ├── test00_000.jpg
	    │   ├── test00_030.jpg
	    │   └── ...
	    └── labels
	        ├── test00_000.txt
	        ├── test00_030.txt
	        └── ...

`data_original` is directory with your images and labels. After running `rotation.py`, rotated images and labels will be stored in `data_rotational`.

## Generate augmented data
	python rotation.py -i DATASET_INPUT

`DATASET_INPUT` is `data_original` in this example.

## Visualize generated images and labels
	python check_label.py -i DATASET_INPUT

`DATASET_INPUT` is `data_rotational` in this example.

