# Create folder for downloaded dataset
if [ ! -d data ]; then
    mkdir data
fi

# Create folder for extracted images
if [ ! -d images ]; then
    mkdir images
fi

# Create symlinks for the 2 repositories used
if [ ! -d caltech-pedestrian-dataset-converter/data ]; then
    cd caltech-pedestrian-dataset-converter && ln -s ../data data && cd ..
fi
if [ ! -d caltech-pedestrian-dataset-extractor/data ]; then
    cd caltech-pedestrian-dataset-extractor && ln -s ../data data && cd ..
fi
if [ ! -d caltech-pedestrian-dataset-extractor/images ]; then
    cd caltech-pedestrian-dataset-extractor && ln -s ../images images && cd ..
fi

cd caltech-pedestrian-dataset-extractor
# Download with the caltech-pedestrian-dataset-extractor/download.sh script
# bash download.sh

# Extract images
npm install
node caltech_pd.js
cd ..

cd caltech-pedestrian-dataset-converter
# Extract annotations
python scripts/convert_annotations.py
cd ..

# Move annotations.json out of data/
mv data/annotations.json ./
