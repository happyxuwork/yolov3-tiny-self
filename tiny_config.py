from easydict import EasyDict as edict
import numpy as np


__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.anchors = np.array([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]])
__C.classes = 3
__C.num = 6
__C.num_anchors_per_layer = 3
__C.batch_size = 8
__C.scratch = False
# __C.names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
__C.names = ['bus','truck','car']
#
# Training options
#
__C.train = edict()

__C.train.ignore_thresh = .5
__C.train.momentum = 0.9
__C.train.decay = 0.0005
__C.train.learning_rate = 0.001
__C.train.max_batches = 50200
__C.train.lr_steps = [10000, 20000]
__C.train.lr_scales = [.1, .1]
__C.train.max_truth = 30
__C.train.mask = np.array([[0, 1, 2], [3, 4, 5]])
#__C.train.image_resized = 416   # { 320, 352, ... , 608} multiples of 32
__C.train.image_width_resized = 512   # { 320, 352, ... , 608} multiples of 32
__C.train.image_hight_resized = 288  # { 320, 352, ... , 608} multiples of 32

#
# image process options
#
__C.preprocess = edict()
__C.preprocess.angle = 0
__C.preprocess.saturation = 1.5
__C.preprocess.exposure = 1.5
__C.preprocess.hue = .1
__C.preprocess.jitter = .3
__C.preprocess.random = 1
