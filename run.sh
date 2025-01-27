#!/bin/zsh
. /opt/csg/spack/share/spack/setup-env.sh
spack env activate cuda
spack load cuda@12.4.0
python models/GoePT/train.py --eval-interval 4 --lr 0.01 --batch-size 24 --epochs 21 --eval-iter 50
