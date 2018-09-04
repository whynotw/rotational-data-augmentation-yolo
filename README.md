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
├── rotational
    ├── images
    │   ├── test00_000.jpg
    │   ├── test00_030.jpg
    │   └── ...
    └── labels
        ├── test00_000.txt
        ├── test00_030.txt
        └── ...

## Generate augmented data
python rotation.py [dataset_name_input]

where [dataset_name_input] is original in this example

## Check images and labels
python rotation.py [dataset_name_output]

where [dataset_name_output] is rotational in this example
