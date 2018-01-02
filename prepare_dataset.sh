#!/bin/sh

python tools/prepare_dataset.py --dataset imagenet_vid --set random --root ~/Deep-Feature-Flow/data/ILSVRC2015 --target ./data_end_test/train.lst
echo Done!
