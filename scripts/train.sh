#!/bin/sh

cd ../exper/

ROOT_DIR=`pwd`/../
DEVICES=0,1

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#CUDA_VISIBLE_DEVICES=$DEVICES python train_frame.py \
#	--num_gpu=2 \
#	--num_workers=4 \
#	--arch=vgg_crsimg_v1 \
#	--epoch=5 \
#	--lr=0.001 \
#	--decay_points=2,3,4 \
#	--batch_size=80 \
#	--crop_size=320 \
#	--loss_local_factor=0.008 \
#	--loss_global_factor=0.001 \
#	--local_seed_num=3 \
#	--resume=False \
#	--dataset=imagenet  \
#	--onehot=False \
#	--save_interval=1 \
#	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
#	--num_classes=1000 \
#	--snapshot_dir="../snapshots/vgg/vgg_crsimg_v1_s3_lfactor008_gfactor0001/"  \
#	--restore_from="../snapshots/vgg/vgg_crsimg_v0/imagenet_epoch_4_glo_step_80075.pth.tar"  \

##	--restore_from="../snapshots/inception3/inception3_v0/imagenet_epoch_4_glo_step_80075.pth.tar"
##	--restore_from=$HOME/xlzhang/torch_models/vgg16-397923af.pth
#
##	--loss_global_factor=0.03 \


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



##CUB
###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#CUDA_VISIBLE_DEVICES=$DEVICES python train_frame_crsimg_center_feat.py \
#	--num_gpu=2 \
#	--num_workers=4 \
#	--arch=inception3_i2c \
#	--epoch=51 \
#	--lr=0.001 \
#	--decay_points=30,40 \
#	--loss_local_factor=0.008 \
#	--loss_global_factor=0.003 \
#	--local_seed_num=3 \
#	--batch_size=80 \
#	--crop_size=224 \
#	--input_size=256 \
#	--save_interval=10 \
#	--resume=False \
#	--dataset=imagenet  \
#	--train_list='../datalist/CUB/train_list.txt' \
#	--img_dir=${ROOT_DIR}/data/CUB_200_2011/images \
#	--num_classes=200 \
#	--snapshot_dir="../cub_snapshots/inception3/inception3_i2c_v1/"  \
#	--restore_from="../islvrc_snapshots/inceptionv3/inceptionv3_i2c_v0/imagenet_init_v0.pth"
#

