#!/bin/bash

#path=$(realpath "${BASH_SOURCE:-$0}")
#DIR_PATH=$(dirname $path)
#Infer_Script="$DIR_PATH/Infer.py"

Infer_Script="./Infer.py"
if [[ "$#" < 1  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CANDLE_DATA_DIR is required"
	    exit -1
fi

CANDLE_DATA_DIR=$1; shift
CMD="python3 ${Infer_Script} $@"



echo "using container "
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
# echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
