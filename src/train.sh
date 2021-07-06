#!/usr/bin/env bash

if [ -e .env ]; then
  source .env
fi

if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

project=EDSR-PyTorch/src
#if [ -d $FASTDIR/git/$project ]; then
#  cd $FASTDIR/git/$project
#elif [ -d /workspace/git/$project ]; then
#  cd /workspace/git/$project
#fi


script=main.py
scale=2
model='unknow'
options=''
pretrain='none'
save='none'

config=config.bin
if [ "$1" != "" ]; then config=$1; fi
if [ -e $config ];
then
  echo "Loading config from $config"
  source $config
fi

if [ "$DELAY" != "" ]; then
  delay=$DELAY
else
  delay=0
fi

if [ "$2" != "" ]; then save=$2; fi

nvidia-smi

python $script --model $model \
  --scale $scale --patch_size $patch_size \
  --dir_data ../../data \
  --save $save \
  --data_train $data_train \
  --data_test $data_test \
  $options

result=$?
echo "python result $result"
if [ "$result" -ne "0" ];
then
  echo $PATH
  echo $LD_LIBRARY_PATH
  which python
  python -V
fi

cd -

#notify-send "cmd finished in $0" "`date`"


