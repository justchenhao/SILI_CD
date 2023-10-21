#!/usr/bin/env bash

gpus=0
config_path=./conf/cd_ours.yaml
checkpoint_root=checkpoints
pretrained=imagenet
splits=train
model_name=ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat
with_wandb=2
scale_mode=3
data_name=LEVIR
scale_ratios=0.25/0.5/0.75/1,0.25
python main_cd.py --scale_mode ${scale_mode}  --data_name ${data_name} --with_wandb $with_wandb --splits ${splits} --scale_ratios ${scale_ratios} --model_name ${model_name} --config_path ${config_path} --gpu_ids ${gpus} --pretrained ${pretrained} --checkpoint_root ${checkpoint_root}
