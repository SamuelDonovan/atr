#!/usr/bin/env bash

# Usage example:
#./training_automation.sh -b 16 -e 5 -i 64 2>&1 | tee training_automation_$(date +"%F_%H").log

if [ $# -lt 3 ]
then
	echo "Not enough arguments supplied"
	echo "Usage:"
	echo "./training_automation.sh b 32 e 10 i 224"
	echo "where b is the batch size"
	echo "where e is the number of epochs"
	echo "where i is the image size in pixels for both height and width"
	echo "m can also be optionally supplied to specify the model to use"
	echo "if m is not specified then all supported models will be tested."
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

declare -a dataTypes=("Photo" "Low" "High"
	"Photo Low High")

# declare -a models=("alexnet" "resnet50" "vit_h_14" 
# 	"vgg11" "densenet201" "maxvit_t"
# 	"swin_v2_t" "efficientnet_v2_s" 
# 	"convnext_tiny" "squeezenet1_1")

declare -a models=("alexnet" 
	"vgg11")

if [ -n "$model" ]; then
	for data in "${dataTypes[@]}"; do
		dataName="${data// /_}"
	 	echo "Only training model: $model"
	 	python3 -m atr --train --test --save --plot --save_name $model_$dataName --batch_size $batch_size --epochs $epochs $image_arg --data $data --model $model
		echo ""
	done
	exit
fi

for data in "${dataTypes[@]}"; do
	dataName="${data// /_}"
	for model in "${models[@]}"; do
		echo python3 -m atr --train --test --save --plot --save_name $model_$dataName --batch_size $batch_size --epochs $epochs --image_size $image_size --data $data --model $model
		python3 -m atr --train --test --save --plot --save_name $model_$dataName --batch_size $batch_size --epochs $epochs $image_arg --data $data --model $model
		echo ""
	done
done