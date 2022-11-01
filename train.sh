#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR

#path=$(realpath "${BASH_SOURCE:-$0}")
#DIR_PATH=$(dirname $path)
#Train_Script="$DIR_PATH/Train.py"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/"
#echo $LD_LIBRARY_PATH

Train_Script="/IGTD/Train.py"
if [[ "$#" < 2  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR are required"
	    exit -1
fi

CUDA_VISIBLE_DEVICES=$1; shift
CANDLE_DATA_DIR=$1; shift
CMD="python3 ${Train_Script} $@"



echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
# echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
