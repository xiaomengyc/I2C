#!/bin/sh

cd ../exper/

ROOT_DIR=`pwd`/../
DEVICES=0,1



##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CUDA_VISIBLE_DEVICES=$DEVICES python train_frame.py \
	--num_gpu=2 \
	--num_workers=4 \
	--arch=inception3_i2c \
	--epoch=5 \
	--lr=0.001 \
	--decay_points=2,3,4 \
	--loss_local_factor=0.008 \
	--local_seed_num=3 \
	--loss_global_factor=0.001 \
	--batch_size=80 \
	--crop_size=224 \
	--input_size=256 \
	--save_interval=1 \
	--onehot=False \
	--resume=False \
	--dataset=imagenet  \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
	--num_classes=1000 \
	--snapshot_dir="../islvrc_snapshots/inceptionv3/inceptionv3_i2c_v1/"  \
	--restore_from="../islvrc_snapshots/inceptionv3/inceptionv3_i2c_v0/imagenet_init_v0.pth.tar"




