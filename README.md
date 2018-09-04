# Rotational Data Augmentation for YOLO

## requirement
	numpy
	opencv-python

## directory structure
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
	python rotation.py <dataset_input>

`<dataset_input>` is `original` in this example.

## Visualize generated images and labels
	python rotation.py <dataset_output>

`<dataset_output>` is `rotational` in this example.

Check the labels are correct or not.

After running this python script successfully, you can move `rotational` to other place where you used to store your data.
