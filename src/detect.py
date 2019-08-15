#! /usr/bin/env python

from __future__ import division

import sensor_msgs
import rospy
import cv2
from cv_bridge import CvBridge


from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import rospkg

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

global debug_show
debug_show = False
global count
count = 0

def callback(img):
    global count
    count += 1
    if count % 3 != 1:
        return
    ts = rospy.get_time()
    header = img.header
    img_gray = CvBridge().imgmsg_to_cv2(img, "mono8")
    height, width = img_gray.shape[:2]

    loader = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img_ = cv2.resize(img_gray, (img_size, img_size))

    image = loader(img_).cuda()
    image = image.view(1, img_size, img_size).expand(3, -1, -1)

    # rospy.loginfo("BF Detection use time {:f}ms".format((rospy.get_time() - ts)*1000))
    
    with torch.no_grad():
        detections = model(image.unsqueeze(0))
        # rospy.loginfo("BF NOMAX use time {:f}ms {}".format((rospy.get_time() - ts)*1000, detections.size()))
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    
    rospy.loginfo_throttle(1.0, "Detection use time {:f}ms".format((rospy.get_time() - ts)*1000))
    if type(detections[0])==torch.Tensor:
        data = detections[0].numpy()
        for i in range(len(data)):
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0].numpy()[i,0:7]
            pt1 = int(x1/img_size*width), int(y1/img_size*height)
            pt2 = int(x2/img_size*width), int(y2/img_size*height)
            if debug_show:
                cv2.rectangle(img_gray, pt1, pt2, (255,0,0), 2)

    if debug_show:
        cv2.imshow("YOLO", img_gray)
        cv2.waitKey(2)

if __name__ == "__main__":
    rospy.init_node('yolov3_ros')

    global image_topic, model_def, weights_path, class_path, conf_thres, nms_thres, img_size
    prefix = rospkg.RosPack().get_path('yolov3_ros') + '/'
    image_topic = rospy.get_param('~image_topic')
    model_def = prefix + rospy.get_param('~model_def')
    weights_path = prefix + rospy.get_param('~weights_path')
    class_path = prefix + rospy.get_param('~class_path')
    conf_thres = rospy.get_param('~conf_thres', 0.8)
    nms_thres = rospy.get_param('~nms_thres', 0.4)
    img_size = rospy.get_param('~img_size', 416)
    debug_show = rospy.get_param('~debug_show', False)

    print image_topic, model_def, weights_path, class_path, conf_thres, nms_thres, img_size
    
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rospy.loginfo("Using device {}".format(device))


    # # Set up model
    global model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        if torch.cuda.is_available():
            model.load_darknet_weights(weights_path)
        else:
            model.load_darknet_weights(weights_path, map_location='cpu')
    else:
        # Load checkpoint weights
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model.eval()  # Set in evaluation mode


    global classes
    classes = load_classes(class_path)  # Extracts class labels from file

    global Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    rospy.Subscriber(image_topic, sensor_msgs.msg.Image, callback, queue_size=1)

    rospy.spin()
