import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from dataset.imagenet_vid import ImageNetVID

dataset_path = '/home/gzhan/Deep-Feature-Flow/data/ILSVRC2015'
data_set = 'VID_train_15frames'
dataset = ImageNetVID(data_set, dataset_path, is_train=False, true_negative_images=True)

dst_path = './tmp'
dataset.make_pair(dst_path)

