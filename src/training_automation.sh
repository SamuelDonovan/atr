#!/usr/bin/env bash

if [ $# -lt 3 ]
then
	echo "Not enough arguments supplied"
	exit 2
fi

while getopts b:e:i:m: option
do 
	case "${option}"
		in
		b)batch_size=${OPTARG};;
		e)epochs=${OPTARG};;
		i)image_size=${OPTARG};;
		m)model=${OPTARG};;
esac
done


echo "Batch Size: $batch_size"
echo "Epochs: $epochs"

if [[ $image_size -eq 0 ]]; then
	image_arg=""
	echo "Running with no image arguments"
else
	image_arg="--image_size $image_size"
	echo "Image arguments: $image_arg"
fi

if [ -n "$model" ]; then
	echo "Only training model: $model"
  python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model $model
else
	echo "No model specified, training all models"
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model alexnet
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model resnet18
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model resnet50
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model vit_h_14
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model vgg11
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model efficientnet_b0
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model densenet121
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model densenet201
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model maxvit_t
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model swin_t
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model swin_v2_t
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model efficientnet_v2_s
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model convnext_tiny
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model squeezenet1_0
	python3 -m atr --train --test --save --batch_size $batch_size --epochs $epochs $image_arg --model squeezenet1_1
fi