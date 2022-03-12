#!/bin/bash
dir=$(pwd)/data
mkdir -p "$dir"
echo "downloading data in $dir"
wget -O "${dir}/ShapeNetSem-8.tar.gz" "https://www.dropbox.com/s/t6thp6np2bhery5/ShapeNetSem-8.tar.gz?dl=1"
wget -O "${dir}/YCB-8.tar.gz" "https://www.dropbox.com/s/kivkqkhmyz8pbtx/YCB-8.tar.gz?dl=1"
wget -O "${dir}/YCB-76.tar.gz" "https://www.dropbox.com/s/84d36gxx018xpg6/YCB-76.tar.gz?dl=1"
echo "extracting data in $dir"
tar -zxvf "${dir}/ShapeNetSem-8.tar.gz" -C "$dir"
tar -zxvf "${dir}/YCB-8.tar.gz" -C "$dir"
tar -zxvf "${dir}/YCB-76.tar.gz" -C "$dir"
echo "finished! data stored in $dir"