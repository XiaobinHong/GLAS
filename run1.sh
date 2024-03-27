#!/usr/bin/env bash
 
echo "Test Begining......"


for dataset in 'amazon-photo' 'cora' 'elliptic' 'fb100' 'ogb-arxiv' 'twitch-e'
do
	for gnn in 'GCN' 'SAGE' 'GAT' 'GPR'
	do
		python main_train.py --dataset $dataset --gnn $gnn --nlayers 2 --hidden 64 --run 10
		wait
		python main_train.py --dataset $dataset --gnn $gnn --nlayers 2 --hidden 128 --run 10
		wait
		python main_train.py --dataset $dataset --gnn $gnn --nlayers 5 --hidden 64 --run 10
		wait
		python main_train.py --dataset $dataset --gnn $gnn --nlayers 5 --hidden 128 --run 10
		wait
	done
done