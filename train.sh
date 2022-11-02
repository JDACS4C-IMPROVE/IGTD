#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

#path=$(realpath "${BASH_SOURCE:-$0}")
#DIR_PATH=$(dirname $path)
#Train_Script="$DIR_PATH/Train.py"

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/"
#echo $LD_LIBRARY_PATH

Train_Script="/usr/local/IGTD/Train.py"
if [[ "$#" < 3  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR CANDLE_CONFIG are required"
	    exit -1
fi

CUDA_VISIBLE_DEVICES=$1; shift
CANDLE_DATA_DIR=$1; shift
CANDLE_CONFIG=$1 ; shift

export CANDLE_DATA_DIR=$CANDLE_DATA_DIR
CMD="python3 ${Train_Script} --config_file $CANDLE_CONFIG"



echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
