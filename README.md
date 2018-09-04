# Rotational Data Augmentation for YOLO

## Requirement
	numpy
	opencv-python

## Directory structure
	.
	├── check_label.py
	├── rotation.py
	├── original
	│   ├── images
	│   │   └── test00.png
	│   └── labels
	│       └── test00.txt
	└── rotational
	    ├── images
	    │   ├── test00_000.jpg
	    │   ├── test00_030.jpg
	    │   └── ...
	    └── labels
	        ├── test00_000.txt
	        ├── test00_030.txt
	        └── ...

Where `original` is directory with your images and labels.

## Generate augmented data
	python rotation.py -i DATASET_INPUT

`DATASET_INPUT` is `original` in this example.

## Visualize generated images and labels
	python rotation.py -i DATASET_INPUT

`DATASET_INPUT` is `rotational` in this example.

