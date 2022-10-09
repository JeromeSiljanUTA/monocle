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
from matplotlib.patches import Rectangle

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
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def show_graph(lined_boxes):
    for box in lined_boxes:
        if(box[0] == 0):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='red')
            plt.scatter(box_coords[2], box_coords[3], c='red')
            plt.imshow(test_img)
        elif(box[0] == 1):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='green')
            plt.scatter(box_coords[2], box_coords[3], c='green')
            plt.imshow(test_img)
        elif(box[0] == 2):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='blue')
            plt.scatter(box_coords[2], box_coords[3], c='blue')
            plt.imshow(test_img)
        elif(box[0] == 3):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='purple')
            plt.scatter(box_coords[2], box_coords[3], c='purple')
            plt.imshow(test_img)
        elif(box[0] == 4):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='yellow')
            plt.scatter(box_coords[2], box_coords[3], c='yellow')
            plt.imshow(test_img)
        elif(box[0] == 5):
            box_coords = box[1]
            plt.scatter(box_coords[0], box_coords[1], c='orange')
            plt.scatter(box_coords[2], box_coords[3], c='orange')
            plt.imshow(test_img)

    plt.savefig(f'result/res_{image_path[16:-4]}_sep{image_path[-4:]}')
    plt.clf()

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

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

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()


    # load data
    for k, image_path in enumerate(image_list):
        thresh_const = 4
        threshold = 0
        prev_endY = 0
        lined_boxes = []
        lined_boxes_index = 0

        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        test_img = img.imread(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # boxes has form [[startX, startY, endX, endY]]
        unsorted_boxes = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[2][1])] for box in bboxes]
        boxes = sorted(unsorted_boxes, key=lambda x: x[1])
        print()

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
        rects = []
        for x in range(max_para + 1):
            para  = [box[1] for box in lined_boxes if box[0] == x]
            min_startX = min([val[0] for val in para])
            min_startY = min([val[1] for val in para])
            max_endX = max([val[2] for val in para])
            max_endY = max([val[3] for val in para])

            print(f'startX: {min_startX}  startY: {min_startY}')
            print(f'endX: {max_endX}    endY: {max_endY}')

            rects.append(para)
            plt.scatter(min_startX, min_startY)
            plt.scatter(max_endX, max_endY)

        plt.show()
        break

        #    plt.scatter(ext_startX, ext_startY)
        #    plt.scatter(ext_endX, ext_endY)
        #    plt.show()
        #    plt.clear()
    


        #show_graph(lined_boxes)

        # save score text
        #filename, file_ext = os.path.splitext(os.path.basename(image_path))
        #mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        #cv2.imwrite(mask_file, score_text)

        #file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    #print("elapsed time : {}s".format(time.time() - t))
