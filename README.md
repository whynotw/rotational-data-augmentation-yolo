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

where `original` is directory with non-rotational images and labels.

## Generate augmented data
	python rotation.py <dataset_name_input>

`<dataset_name_input>` is `original` in this example.

## Check images and labels
	python rotation.py [dataset_name_output]

`<dataset_name_output>` is `rotational` in this example.

After running this python script, you can `mv` `rotational` to other place you used to restore data.
