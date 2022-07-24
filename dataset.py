# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:02:28 2022

@author: Paul
@file: dataset.py
@dependencies:
    env pt3.7
    python 3.7.13
    numpy >= 1.19.2
    torch >= 1.7.1
    torchvision >= 0.8.2
    pandas >= 1.2.1
    Pillow >= 8.1.0 

@references:
    Redmon, Joseph and Farhadi, Ali, YOLOv3: An Incremental Improvement, April 8, 2018. (https://doi.org/10.48550/arXiv.1804.02767)
    Ayoosh Kathuria, Whats new in YOLO v3?, April, 23, 2018. (https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
    Sanna Persson, YOLOv3 from Scratch, Mar 21, 2021. (https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)

Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config # for hyper-parameter tuning stuffs
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

from PIL import Image, ImageFile # use pillow to load the images
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

# set to Ture so that we don't get any errors when we loading the images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# create our own custom YOLODataset, first we're going to inherit from torch.utils.data.Dataset
class YOLODataset(Dataset):
    def __init__(self, 
        csv_file, 
        img_dir, 
        label_dir, 
        anchors, 
        image_size=16, # image_size=416, 
        S=[2, 4, 8],   # S=[13, 26, 52], 
        C=1, # C=20, 
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file) # "D:/Datasets/RD_maps/train.csv"
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size #
        self.transform = transform   #
        self.S = S                   # grid size for each scale
        # put all the anchors together for all 3 scales, each scale have 3 anchors store in anchors[i]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        self.num_anchors = self.anchors.shape[0] # 9
        self.num_anchors_per_scale = self.num_anchors // 3 # divided by 3, because we're assuming that we have 3 scales
        self.C = C                   # number of classes
        self.ignore_iou_thresh = 0.5 # if the iou is greater than 0.5, then we're going to ignore the prediction for that one

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # get the index-th data, in the csv files, data are structured as indxe.jpg,index.txt, 
        # so (indxe, 0) get us the image and (index, 1) get us the label

        # get the label directory path (self.label_dir), then get the csv file name (self.annotations), then get the .txt file 
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # on the second column (which is 1)

        # after getting the .txt file path, we then load the .txt file which is delimited by space, and we set the returned array 
        # will have at least ndmin=2 dimensions, the original label is [class, x, y, w, h]

        # later on, we're going to use the albumentations library for augmentation (to augment the bounding boxes, images), but it 
        # requires the class label to be last in the bounding box, which is [x, y, w, h, class], we use np.roll() to do so
        # np.roll(array, shift=4, axis=1) will roll the array to the right by 4 on the x-axis (axis=1 means horizontally)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist()

        # get the image directory path (self.img_dir), then get the csv file name (self.annotations), then get the .jpg file
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # on the first column (which is 0)
        # after getting the .jpg file path, we then load the .jpg file and we're going to convert it into RGB, for using 
        # albumentations, we need to also make sure that it's a np array
        image = np.array(Image.open(img_path).convert("RGB"))

        # check if we have some transforms
        if self.transform:
            # send in the image and bounding boxes that way if we rotate the image, the bounding boxes are still gonna be correct
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # below assumes 3 scale predictions (as paper) and same num of anchors per scale, which is 3
        # self.num_anchors // 3 = 9 // 3 = 3 = num of scale predictions, S = grid size for each scale
        # 6 stands for [P(Object), x, y, w, h, class] which is in total 6 values
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        print("len(targets): ", len(targets)) # len(targets) = 3

        # then we go through each of the bounding boxes, each of the 3 scales should do a prediction and we need to assign 
        # which anchor should be responsible at which particular cells
        for box in bboxes:
            # each box contain [x, y, w, h, class]
            # now we have the targets for all the different scales, we want to go through each of our bounding box in the image
            # and we want to assign which anchor should be responsible , which cell should be responsible for all 3 diff scales

            # so how we assign which anchor is responsible? we check which one has the highest iou
            # send in the [width, height] and the pre-defined anchors (self.anchors) to calculate the ious for the particular 
            # box and for all the anchors, which are 9 diff ones
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) 

            # then we want to check which anchor is the best? we use torch.argsort() along y-axis (dim=0) with descending order, 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # so the first one, first indicies is the best anchor
            x, y, width, height, class_label = box  # we can also take out the [x, y, width, height] from the box
            has_anchor = [False, False, False]      # we're going to make sure that each of 3 scales should have 1 anchor

            # then we go through each anchors, starting from the best one till the worse one 
            for anchor_idx in anchor_indices:
                # 
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    # PASCAL VOC "D:/Datasets/PASCAL_VOC/train.csv", "D:/Datasets/PASCAL_VOC/images", "D:/Datasets/PASCAL_VOC/labels",
    # MS COCO    "COCO/train.csv", "COCO/images/images/", "COCO/labels/labels_new/"
    dataset = YOLODataset(
        "D:/Datasets/RD_maps/train.csv", # csv_file 
        "D:/Datasets/RD_maps/images",    # img_dir 
        "D:/Datasets/RD_maps/labels",    # label_dir 
        S=[2, 4, 8], # S=[13, 26, 52],   
        anchors=anchors, 
        transform=transform, 
    )
    S = [2, 4, 8] # S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(f"anchor.shape: {anchor.shape}")
            print(f"y[{i}].shape: {y[i].shape}")
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(f"boxes: {boxes}")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)

        # e.g. 
        # anchor.shape: torch.Size([3, 2])
        # y[0].shape: torch.Size([1, 3, 2, 2, 6])
        # anchor.shape: torch.Size([3, 2])
        # y[1].shape: torch.Size([1, 3, 4, 4, 6])
        # anchor.shape: torch.Size([3, 2])
        # y[2].shape: torch.Size([1, 3, 8, 8, 6])
        # boxes: [[0.0, 1.0, 0.125, 0.6875, 0.125, 0.125], [0.0, 1.0, 0.125, 0.6875, 0.125, 0.125], [0.0, 1.0, 0.125, 0.6875, 0.125, 0.125]]


if __name__ == "__main__":
    test()
