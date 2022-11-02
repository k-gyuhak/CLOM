#!/bin/sh
t=0

# Representation train
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--nproc_per_node=2 \
train.py \
--dataset cifar100_10t \
--model resnet18 \
--mode sup_simclr_CSI \
--batch_size 64 \
--epoch 700 \
--t $t \
--lamb0 1.5 \
--lamb1 1.0

# Linear layer train
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--mode sup_CSI_linear \
--dataset cifar100_10t \
--model resnet18 \
--batch_size 128 \
--epoch 100 \
--t $t

# ACCURACY & ECE
CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode test_marginalized_acc \
--dataset cifar100_10t \
--model resnet18 \
--t $t \
--all_dataset \
--printfn 'til.txt'

CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil \
--dataset cifar100_10t \
--model resnet18 \
--batch_size 128 \
--cil_task $t \
--all_dataset \
--printfn "cil results.txt"

CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil_pre \
--dataset cifar100_10t \
--model resnet18 \
--batch_size 32 \
--cil_task $t \
--printfn "calibration.txt" \
--adaptation_lr 0.01 \
--weight_decay=0

for t in 1 2 3 4 5 6 7 8 9
do
	# Representation train
	CUDA_VISIBLE_DEVICES=0,1 \
	python -m torch.distributed.launch \
	--nproc_per_node=2 \
	train.py \
	--dataset cifar100_10t \
	--model resnet18 \
	--mode sup_simclr_CSI \
	--batch_size 64 \
	--epoch 700 \
	--t $t \
	--lamb0 1.5 \
	--lamb1 1.0

	# linear layer train
	CUDA_VISIBLE_DEVICES=0 \
	python train.py \
	--mode sup_CSI_linear \
	--dataset cifar100_10t \
	--model resnet18 \
	--batch_size 128 \
	--epoch 100 \
	--t $t

	# ACCURACY & ECE
	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode test_marginalized_acc \
	--dataset cifar100_10t \
	--model resnet18 \
	--t $t \
	--all_dataset \
	--printfn 'til.txt'

	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode cil \
	--dataset cifar100_10t \
	--model resnet18 \
	--batch_size 128 \
	--cil_task $t \
	--all_dataset \
	--printfn "cil results.txt"

	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode cil_pre \
	--dataset cifar100_10t \
	--model resnet18 \
	--batch_size 32 \
	--cil_task $t \
	--printfn "calibration.txt" \
	--adaptation_lr 0.01 \
	--weight_decay=0
done