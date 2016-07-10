# Create dataset folder, if need be
# This could be a symlink to another location
if [ ! -d dataset ]; then
    mkdir dataset
fi

# Create folder for downloaded dataset
if [ ! -d dataset/data ]; then
    mkdir dataset/data
fi

# Create folder for extracted images
if [ ! -d dataset/images ]; then
    mkdir dataset/images
fi

# Create symlinks for the 2 repositories used
if [ ! -d caltech-pedestrian-dataset-converter/data ]; then
    cd caltech-pedestrian-dataset-converter && ln -s ../dataset/data data && cd ..
fi
if [ ! -d caltech-pedestrian-dataset-extractor/data ]; then
    cd caltech-pedestrian-dataset-extractor && ln -s ../dataset/data data && cd ..
fi
if [ ! -d caltech-pedestrian-dataset-extractor/images ]; then
    cd caltech-pedestrian-dataset-extractor && ln -s ../dataset/images images && cd ..
fi

cd caltech-pedestrian-dataset-extractor
# Download with the caltech-pedestrian-dataset-extractor/download.sh script
bash download.sh

# Extract images
npm install
node caltech_pd.js
cd ..

cd caltech-pedestrian-dataset-converter
# Extract annotations
python scripts/convert_annotations.py
cd ..

# Move annotations.json out of data/
mv dataset/data/annotations.json dataset/
