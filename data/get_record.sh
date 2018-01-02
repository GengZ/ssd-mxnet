#!/bin/sh

python ~/mxnet/tools/im2rec.py ./ ~/Deep-Feature-Flow/data/ILSVRC2015 --shuffle=1 --pack-label=1 --num-thread=16 > record.log 2>&1 &
