#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1
NET_FINAL=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="exps/${EXP_NAME}/logs/test_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -n "$NET_FINAL" ]; 
then
   NET_FINAL=`find exps/${EXP_NAME}/output -type f -ipath "*${NET_FINAL}*.caffemodel" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
else
   NET_FINAL=`find exps/${EXP_NAME}/output -type f -ipath '*.caffemodel' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
fi

echo ${NET_FINAL}

time ./tools/test_net.py --gpu 0 \
  --net ${NET_FINAL} \
  --cfg exps/${EXP_NAME}/config.yml \
  --exp_dir ${EXP_NAME} \
  --datasets ${EXTRA_ARGS}
