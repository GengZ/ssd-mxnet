#!/bin/sh

python tools/prepare_dataset.py --dataset imagenet_vid --set test --root ~/Deep-Feature-Flow/data/ILSVRC2015 --target ./rec_data/test.lst --shuffle False
python tools/prepare_dataset.py --dataset imagenet_vid --set random --root ~/Deep-Feature-Flow/data/ILSVRC2015 --target ./rec_data/train.lst
echo prepare data done
cp ./rec_data/* /home/gzhan/data/vid_rec/
echo copy data
cd ~/git/mx_rfcn/pdet
rm -r ~/git/mx_rfcn/pdet/output/vid_debut
echo clear log
python ./allegro.py --cfg ./vid_test.yaml
echo run
cd -
echo Done!
