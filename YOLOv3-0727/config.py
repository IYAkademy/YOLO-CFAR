# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:01:48 2022

@author: Paul
@file: config.py
@dependencies:
    env pt3.7
    python 3.7.13
    numpy >= 1.19.2
    torch >= 1.7.1
    torchvision >= 0.8.2
    albumentations >= 0.5.2

"""

import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything

DATASET = 'D:/Datasets/RD_maps' # 'D:/Datasets/PASCAL_VOC', 'D:/Datasets/RD_maps'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# seed_everything()  # If you want deterministic behavior

NUM_WORKERS = 1      # 4
BATCH_SIZE = 16      # 32
IMAGE_SIZE = 416     # 416
NUM_CLASSES = 1     # PASCAL VOV has 20 classes, MS COCO has 80 classes
LEARNING_RATE = 1e-4 # 3e-5
WEIGHT_DECAY = 0     # 1e-4, 5e-4
NUM_EPOCHS = 10      # 1000
CONF_THRESHOLD = 0.4 # 0.6
MAP_IOU_THRESH = 0.5 
NMS_IOU_THRESH = 0.45 
stride = [32, 16, 8] 
S = [IMAGE_SIZE // stride[0], IMAGE_SIZE // stride[1], IMAGE_SIZE // stride[2]] # [13, 26, 52]
# S = [IMAGE_SIZE // 8, IMAGE_SIZE // 4, IMAGE_SIZE // 2] # [2, 4, 8]
PIN_MEMORY = False # True
LOAD_MODEL = False # True
SAVE_MODEL = True
# "D:/Datasets/PASCAL_VOC/checkpoint.pth.tar", "D:/Datasets/RD_maps/checkpoint.pth.tar"
CHECKPOINT_FILE = "D:/Datasets/RD_maps/checkpoint.pth.tar" # 
IMG_DIR = DATASET + "/scaled_colors/"   # "/images/"
LABEL_DIR = DATASET + "/labels/" # "/labels/"

# how we handle the anchor boxes? we will specify the anchor boxes in the following manner as a list of lists 
# of tuples, where each tuple corresponds to the width and the height of a anchor box relative to the image size 
# and each list grouping together three tuples correspond to the anchors used on a specific prediction scale

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # largest anchor boxes
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # medium anchor boxes
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # small anchor boxes
# ]  
# note these anchors above are the ones used in the original paper and have been rescaled to be between [0, 1]

ANCHORS = [
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
]

scale = 1.0 # 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=10, p=0.4, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# ValueError: Expected y_max for bbox (0.375, 0.9375, 0.5, 1.0625, 0.0) to be in the range [0.0, 1.0], got 1.0625.
train_transforms, test_transforms = None, None # transform would cause unknown error

CLASSES = [
    "target"
]

# PASCAL_CLASSES
# CLASSES = [
#     "aeroplane",
#     "bicycle",
#     "bird",
#     "boat",
#     "bottle",
#     "bus",
#     "car",
#     "cat",
#     "chair",
#     "cow",
#     "diningtable",
#     "dog",
#     "horse",
#     "motorbike",
#     "person",
#     "pottedplant",
#     "sheep",
#     "sofa",
#     "train",
#     "tvmonitor"
# ]

# COCO_LABELS
# CLASSES = [
#     'person',
#     'bicycle',
#     'car',
#     'motorcycle',
#     'airplane',
#     'bus',
#     'train',
#     'truck',
#     'boat',
#     'traffic light',
#     'fire hydrant',
#     'stop sign',
#     'parking meter',
#     'bench',
#     'bird',
#     'cat',
#     'dog',
#     'horse',
#     'sheep',
#     'cow',
#     'elephant',
#     'bear',
#     'zebra',
#     'giraffe',
#     'backpack',
#     'umbrella',
#     'handbag',
#     'tie',
#     'suitcase',
#     'frisbee',
#     'skis',
#     'snowboard',
#     'sports ball',
#     'kite',
#     'baseball bat',
#     'baseball glove',
#     'skateboard',
#     'surfboard',
#     'tennis racket',
#     'bottle',
#     'wine glass',
#     'cup',
#     'fork',
#     'knife',
#     'spoon',
#     'bowl',
#     'banana',
#     'apple',
#     'sandwich',
#     'orange',
#     'broccoli',
#     'carrot',
#     'hot dog',
#     'pizza',
#     'donut',
#     'cake',
#     'chair',
#     'couch',
#     'potted plant',
#     'bed',
#     'dining table',
#     'toilet',
#     'tv',
#     'laptop',
#     'mouse',
#     'remote',
#     'keyboard',
#     'cell phone',
#     'microwave',
#     'oven',
#     'toaster',
#     'sink',
#     'refrigerator',
#     'book',
#     'clock',
#     'vase',
#     'scissors',
#     'teddy bear',
#     'hair drier',
#     'toothbrush'
# ]
