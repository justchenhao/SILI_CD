#!/usr/bin/env bash


config_path=./conf/cd_ours.yaml
data_name=LEVIR
model_name=ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat
checkpoint_dir='checkpoints/ours_levir1x'
python main_eval.py --model_name ${model_name} --config_path ${config_path} --data_name ${data_name} --checkpoint_dir ${checkpoint_dir}
