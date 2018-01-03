#!/bin/sh

python tools/prepare_dataset.py --dataset imagenet_vid --set test --root ~/vid_data/ILSVRC2015 --target ./rec_data/test.lst --shuffle False
python tools/prepare_dataset.py --dataset imagenet_vid --set random --root ~/vid_data/ILSVRC2015 --target ./rec_data/train.lst
echo Done!
