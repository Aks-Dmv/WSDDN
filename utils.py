import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

#TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    confident_bboxes = bounding_boxes[confidence_score>threshold]
    scores_of_bboxes = confidence_score[confidence_score>threshold]
    boxes = []
    scores = []
    
    x1 = confident_bboxes[:,0]
    y1 = confident_bboxes[:,1]
    x2 = confident_bboxes[:,2]
    y2 = confident_bboxes[:,3]
    
    areas = (y2 - y1 + 1)*(x2 - x1 + 1)
    sorted_indexes = torch.argsort(y2)
    
    while(sorted_indexes.shape[0]>0):
        i = sorted_indexes[-1]
        boxes.append(confident_bboxes[i])
        scores.append(scores_of_bboxes[i])
        
        max_x1 = torch.max(x1[i], x1[sorted_indexes[:-1]])
        max_y1 = torch.max(y1[i], y1[sorted_indexes[:-1]])
        min_x2 = torch.min(x2[i], x2[sorted_indexes[:-1]])
        min_y2 = torch.min(y2[i], y2[sorted_indexes[:-1]])
        
        w = torch.max(torch.zeros(min_x2.shape).cuda(), min_x2 - max_x1 + 1)
        h = torch.max(torch.zeros(min_y2.shape).cuda(), min_y2 - max_y1 + 1)
        
        overlap = (w * h) / areas[sorted_indexes[:-1]]
        
        sorted_indexes = sorted_indexes[:-1]
        sorted_indexes = sorted_indexes[overlap<0.3]


    return boxes, scores


#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """

    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    # print(classes)
    # print(bbox_coordinates[0][0])
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list


