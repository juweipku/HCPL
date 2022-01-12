#!/bin/bash

gpu_num=0

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --public_splits --dataset cora --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --public_splits --dataset citeseer --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --public_splits --dataset pubmed --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --dataset computers --percentiles_holder 1 --uncertainty_percentiles 0.01  --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --dataset photo --percentiles_holder 1 --uncertainty_percentiles 0.01  --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done

for i in $(seq 1 5)
do
  CUDA_VISIBLE_DEVICES=$gpu_num python train.py --dataset cs --percentiles_holder 1 --uncertainty_percentiles 0.01  --hidden 256 --log_file 'main_log.txt' --expt_name 'main'
done