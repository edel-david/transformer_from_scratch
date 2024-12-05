#!/bin/zsh
. /opt/csg/spack/share/spack/setup-env.sh
spack env activate cuda
spack load cuda@12.4.0
python info.py
python models/GoePT/model.py --eval-interval 3 --lr 0.7 --batch-size 16
