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


class SwarmDetector:
    def __init__(self):
        prefix = rospkg.RosPack().get_path('yolov3_ros') + '/'
        image_topic = rospy.get_param('~image_topic')
        model_def = prefix + rospy.get_param('~model_def')
        weights_path = prefix + rospy.get_param('~weights_path')
        class_path = prefix + rospy.get_param('~class_path')
        self.conf_thres = conf_thres = rospy.get_param('~conf_thres', 0.95) # Upper than some value
        self.nms_thres = nms_thres = rospy.get_param('~nms_thres', 0.1)
        img_size = self.img_size = rospy.get_param('~img_size', 416)
        self.debug_show = rospy.get_param('~debug_show', False)

        print(image_topic, model_def, weights_path, class_path, conf_thres, nms_thres, img_size)
        
        
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rospy.loginfo("Using device {}".format(device))

        self.bridge = CvBridge()


        # # Set up model
        self.model = model = Darknet(model_def, img_size=img_size).to(device)

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


        classes = load_classes(class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        rospy.Subscriber(image_topic, sensor_msgs.msg.Image, self.gray_callback, queue_size=1)
        # rospy.Subscriber("/camera/depth/image_rect_raw", sensor_msgs.msg.Image, self.depth_callback, queue_size=1)

        self.depth_img = None
        self.depths = None

    def depth_callback(self, depth_img):
        # print("DEPTH", depth_img.header.stamp)
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")


        self.depths = cv_image_array = np.array(self.depth_img, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        self.depthimg = cv_image_norm
        cv2.imshow("Image from my node", self.depthimg)
        cv2.waitKey(1)

    def get_depth(self, bboxes):
        if self.depths is None:
            return
        
        count = 0
        for pt1, pt2 in bboxes:
            crop_img = self.depths[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            w = pt2[0] - pt1[0]
            h = pt2[1] - pt1[1]
            if self.debug_show:
                crop_img_dis = cv2.resize(crop_img, (w*5, h*5))
                cv2.imshow("cropped {}".format(count), crop_img_dis)
                cv2.waitKey(10)
            crop_depths = np.reshape(crop_img, (w*h))
            # print(crop_depths)
            # plt.figure(count, figsize=(10, 10))
            # plt.cla()
            # plt.hist(crop_depths, bins=50)
            # plt.pause(0.02)


            count += 1


    def gray_callback(self, img):
        # print("GRAY", img.header.stamp)
        ts = rospy.get_time()
        img_gray = self.bridge.imgmsg_to_cv2(img, "mono8")
        height, width = img_gray.shape[:2]

        loader = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_size = self.img_size
        img_ = cv2.resize(img_gray, (img_size, img_size))

        image = loader(img_).cuda()
        image = image.view(1, img_size, img_size).expand(3, -1, -1)

        
        with torch.no_grad():
            detections = self.model(image.unsqueeze(0))
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        
        rospy.loginfo_throttle(1.0, "Detection use time {:f}ms".format((rospy.get_time() - ts)*1000))
        bboxes = []
        if type(detections[0])==torch.Tensor:
            data = detections[0].numpy()
            for i in range(len(data)):
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0].numpy()[i,0:7]
                if x2 - x1 < 0.03*img_size or  y2 - y1 < 0.02*img_size:
                    continue

                pt1 = int(x1/img_size*width), int(y1/img_size*height)
                pt2 = int(x2/img_size*width), int(y2/img_size*height)
                bboxes.append((pt1, pt2))
                if self.debug_show:
                    cv2.rectangle(img_gray, pt1, pt2, (255,0,0), 2)

        if self.debug_show:
            _img = cv2.resize(img_gray, (1280, 960))
            cv2.imshow("YOLO", _img)
            cv2.waitKey(2)

        # self.get_depth(bboxes)


if __name__ == "__main__":
    rospy.init_node('yolov3_ros')
    sd = SwarmDetector()
    plt.ion()
    rospy.spin()
