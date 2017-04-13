#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

if [ ! -d "exps/${EXP_NAME}/logs" ]; then
    mkdir "exps/${EXP_NAME}/logs"
fi

LOG="exps/${EXP_NAME}/logs/train_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python3 ./tools/train_net.py --gpu 0 \
  --cfg exps/${EXP_NAME}/config.yml \
  --exp_dir ${EXP_NAME} \
  ${EXTRA_ARGS}

HTML_OUTPUT="exps/${EXP_NAME}/output/train_loss_`date +'%Y-%m-%d_%H-%M-%S'`.html"

./tools/plot_train_curve.py --log ${LOG} --output ${HTML_OUTPUT}