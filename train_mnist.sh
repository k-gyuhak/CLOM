#!/bin/sh
t=0

# Representation train
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset mnist \
--model mlp \
--mode sup_simclr_CSI \
--batch_size 256 \
--epoch 700 \
--t $t \
--amp \
--lamb0 0.25 \
--lamb1 0.1

# Linear layer train
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--mode sup_CSI_linear \
--dataset mnist \
--model mlp \
--batch_size 256 \
--epoch 100 \
--t $t

# ACCURACY & ECE
CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode test_marginalized_acc \
--dataset mnist \
--model mlp \
--t $t \
--all_dataset \
--printfn 'til.txt'

CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil \
--dataset mnist \
--model mlp \
--batch_size 256 \
--cil_task $t \
--all_dataset \
--printfn "cil results.txt"

CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil_pre \
--dataset mnist \
--model mlp \
--batch_size 32 \
--cil_task $t \
--printfn "calibration.txt" \
--adaptation_lr 0.01 \
--weight_decay=0

for t in 1 2 3 4
do
	# Representation train
	CUDA_VISIBLE_DEVICES=0 \
	python train.py \
	--dataset mnist \
	--model mlp \
	--mode sup_simclr_CSI \
	--batch_size 256 \
	--epoch 700 \
	--t $t \
	--amp \
	--lamb0 0.25 \
	--lamb1 0.1

	# linear layer train
	CUDA_VISIBLE_DEVICES=0 \
	python train.py \
	--mode sup_CSI_linear \
	--dataset mnist \
	--model mlp \
	--batch_size 256 \
	--epoch 100 \
	--t $t

	# ACCURACY & ECE
	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode test_marginalized_acc \
	--dataset mnist \
	--model mlp \
	--t $t \
	--all_dataset \
	--printfn 'til.txt'

	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode cil \
	--dataset mnist \
	--model mlp \
	--batch_size 256 \
	--cil_task $t \
	--all_dataset \
	--printfn "cil results.txt"

	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode cil_pre \
	--dataset mnist \
	--model mlp \
	--batch_size 32 \
	--cil_task $t \
	--printfn "calibration.txt" \
	--adaptation_lr 0.01 \
	--weight_decay=0
done