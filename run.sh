#!/bin/zsh
. /opt/csg/spack/share/spack/setup-env.sh
spack env activate cuda
spack load cuda@12.4.0
python info.py
python models/GoePT/model.py --eval-interval 10 --lr 0.02 --batch-size 24 --epochs 200
