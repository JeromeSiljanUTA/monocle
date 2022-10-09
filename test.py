"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib import image as img
import pytesseract

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# parser
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, poly, refine_net=None):
    t0 = time.time()

    # resize                                                                    canvas_size
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)


    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    #print(boxes)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def read_img(image_path):
    thresh_const = 4
    threshold = 0
    prev_endY = 0
    lined_boxes = []
    lined_boxes_index = 0

    print(f'Test image {image_path}')
    image = imgproc.loadImage(image_path)
    test_img = img.imread(image_path)

    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly, refine_net)

    # boxes has form [[startX, startY, endX, endY]]
    unsorted_boxes = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[2][1])] for box in bboxes]
    boxes = sorted(unsorted_boxes, key=lambda x: x[1])

    for index, box in enumerate(boxes):
        if(index == 0):
            threshold = thresh_const + abs(box[1] - box[3])
        elif(abs(prev_endY - box[1]) > threshold):
            lined_boxes_index += 1

        lined_boxes.append([lined_boxes_index, box])
        prev_endY = box[1]
    print()
    max_para = max([box[0] for box in lined_boxes])

    plt.imshow(test_img)
    message = []
    for x in range(max_para + 1):
        para  = [box[1] for box in lined_boxes if box[0] == x]
        min_startX = min([val[0] for val in para])
        min_startY = min([val[1] for val in para])
        max_endX = max([val[2] for val in para])
        max_endY = max([val[3] for val in para])

        orig = cv2.imread(image_path)
        crop = orig[min_startY:max_endY, min_startX:max_endX]

        ocr_string = pytesseract.image_to_string(crop)
        message.append(ocr_string)

    print(f'{image_path} message: {message}')

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    #print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    read_img('training-strips/cartoon2.PNG')
