#!/bin/zsh
. /opt/csg/spack/share/spack/setup-env.sh
spack env activate cuda
spack load cuda@12.4.0
python models/GoePT/model.py --eval-interval 4 --lr 0.04 --batch-size 24 --epochs 100 --eval-iter 50
