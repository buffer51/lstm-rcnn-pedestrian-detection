# Data extraction

These scripts download and extract the data from the Caltech Dataset.

**caltech-pedestrian-dataset-converter** contains Python scripts to download
the dataset, and extract annotations. The script for extracting images
doesn't work (functionnality removed from OpenCV).

**caltech-pedestrian-dataset-extractor** is built with NodeJS and extracts
images from the dataset.

To aggregate images from a sequence into a video, you can use **FFMPEG**:
```
ffmpeg -framerate 30 -i img%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
```
