# Data extraction

## Process

The `download_parse.sh` script download & extracts the images and annotations,
using the following two repositories:

- **caltech-pedestrian-dataset-converter** contains Python scripts to download
the dataset, and extract annotations. The script for extracting images
doesn't work (functionnality removed from OpenCV).

- **caltech-pedestrian-dataset-extractor** is built with NodeJS and extracts
images from the dataset.

Simply run:
```
./download_parse.sh
```

## Requirements

The first repository requires **scipy** to be installed, and the second
requires both **npm** and **NodeJS** (*v4* at least).

## Making videos

To aggregate images from a sequence into a video, you can use **FFMPEG**:
```
ffmpeg -framerate 30 -i %d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
```
