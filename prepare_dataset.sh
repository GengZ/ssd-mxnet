#!/bin/sh

# python tools/prepare_dataset.py --dataset imagenet_vid --set test --root ~/data/vid_data --target ./rec_data/test.lst --shuffle False
python tools/prepare_dataset.py --dataset imagenet_vid --set det_train --root ~/data/vid_data --target ./vid_rec/train.lst
# echo prepare data done
# cp ./rec_data/* ~/data/vid_rec/
# echo copy data
# cd ~/git/mx_rfcn/pdet
# rm -r ~/git/mx_rfcn/pdet/output/vid_debut
# echo clear log
# python ./allegro.py --cfg ./vid_test.yaml
# echo run
# cd -
# echo Done!
