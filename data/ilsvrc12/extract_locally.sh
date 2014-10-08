#!/bin/bash

set -x

if [ $(df -h /scr/ | tail -n1 | py -x 'x.split()[3][:-1]') -le 200 ]; then
    echo "You need to free up some space first"
    exit
fi

tar_dir=/scail/group/deeplearning/sail-deep-gpu/brodyh/data/ILSVRC_2012
extract_dir=/scr/brodyh/ilsvrc12

# extract training images
mkdir -p $extract_dir/train
tar -C $extract_dir/train/ -xvf $tar_dir/ILSVRC2012_img_train.tar

for tar_file in $(ls $extract_dir/train/*tar); do
    tar_dir=${tar_file%.*}
    mkdir -p $tar_dir
    tar -C $tar_dir/ -xf $tar_file
    rm $tar_file
done

# extract validation images
mkdir -p $extract_dir/val
tar -C $extract_dir/val/ -xvf $tar_dir/ILSVRC2012_img_val.tar
