#!/bin/bash

# Define paths
mkdir -p data/gta/ann_dir/train  data/gta/ann_dir/test  data/gta/ann_dir/val

SRC_DIR="/home/cbtil3/Downloads/REFUGE/train/Masks"
cp -r $SRC_DIR/* data/gta/ann_dir/train
SRC_DIR="/home/cbtil3/Downloads/REFUGE/test/Masks"
cp -r $SRC_DIR/* data/gta/ann_dir/test
SRC_DIR="/home/cbtil3/Downloads/REFUGE/val/Masks"
cp -r $SRC_DIR/* data/gta/ann_dir/val

mkdir -p data/gta/img_dir/train  data/gta/img_dir/test  data/gta/img_dir/val

SRC_DIR="/home/cbtil3/Downloads/REFUGE/train/Images"
cp -r $SRC_DIR/* data/gta/img_dir/train
SRC_DIR="/home/cbtil3/Downloads/REFUGE/test/Images"
cp -r $SRC_DIR/* data/gta/img_dir/test
SRC_DIR="/home/cbtil3/Downloads/REFUGE/val/Images"
cp -r $SRC_DIR/* data/gta/img_dir/val


echo "Done moving data"
