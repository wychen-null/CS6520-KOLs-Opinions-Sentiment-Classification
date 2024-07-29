#!/bin/bash
epochs=(6 8 10 12)
lora_r=(4 8)
dropout=(0.15)

for (( epoch=0; epoch<4; epoch=epoch+1 )) do
	for ((r=0; r<2; r=r+1 )) do
		for (( d=0; d<1; d=d+1 )) do
		  python llama2_classification_mlp.py --lora_r ${lora_r[$r]} --epochs ${epochs[$epoch]} --dropout ${dropout[$d]} --pretrained_ckpt /mnt/ai2lab/yinqiaoli/tmpp-user/yuchen/model/Llama-2-7b-hf & wait
		done
	done
done
