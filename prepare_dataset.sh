#!/bin/sh

python tools/prepare_dataset.py --dataset imagenet_vid --set random --root ~/vid_data/ILSVRC2015 --target ./data/train.lst
