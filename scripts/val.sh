#!/bin/sh

cd ../exper/

ROOT_DIR=`pwd`/../
DEVICES=0

CUDA_VISIBLE_DEVICES=$DEVICES python val_frame.py \
    --arch=inception3_i2c \
	--batch_size=1 \
	--input_size=224 \
	--num_gpu=1 \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
	--dataset=imagenet \
	--num_classes=1000 \
	--save_atten_dir="" \
	--snapshot_dir=""  

	# --tencrop=True \

