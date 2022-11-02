#!/bin/bash

# arg 1 CANDLE_DATA_DIR
# arg 2 CANDLE_CONFIG

#path=$(realpath "${BASH_SOURCE:-$0}")
#DIR_PATH=$(dirname $path)
#Infer_Script="$DIR_PATH/Infer.py"

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/"
#echo $LD_LIBRARY_PATH

Infer_Script="/usr/local/IGTD/Infer.py"
if [[ "$#" < 2  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CANDLE_DATA_DIR CANDLE_CONFIG are required"
	    exit -1
fi

CANDLE_DATA_DIR=$1; shift
CANDLE_CONFIG=$1; shift

export CANDLE_DATA_DIR=$CANDLE_DATA_DIR
CMD="python3 ${Infer_Script} $@"



echo "using container "
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
# echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
