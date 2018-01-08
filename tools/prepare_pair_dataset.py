import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from dataset.imagenet_vid import ImageNetVID

# dataset_path = '/home/gzhan/Deep-Feature-Flow/data/ILSVRC2015'
dataset_path = '/data/home/v-gezhan/data/vid_data'
# data_set = 'VID_train_15frames'
data_set = 'DET_train_30classes'
dataset = ImageNetVID(data_set, dataset_path, is_train=False, true_negative_images=True)

dst_path = './tmp'
# dst_path = '/data/home/v-gezhan/my_remote_disk/dataset_test'
dataset.make_pair(dst_path)

