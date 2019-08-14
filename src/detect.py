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
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

pub = rospy.Publisher('/yolov3_ros/image', sensor_msgs.msg.Image, queue_size=10)

def callback(img):
    header = img.header
    img_gray = CvBridge().imgmsg_to_cv2(img, "mono8")
    height, width = img_gray.shape[:2]
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_ = cv2.resize(img_rgb, (img_size, img_size))
    img_ = torch.from_numpy(img_.transpose((2, 0, 1)))
    image = img_.float().div(255).unsqueeze(0)
    input_img = Variable(image.type(Tensor))
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    
    if type(detections[0])==torch.Tensor:
        data = detections[0].numpy()
        for i in range(len(data)):
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0].numpy()[i,0:7]
            pt1 = int(x1/img_size*width), int(y1/img_size*height)
            pt2 = int(x2/img_size*width), int(y2/img_size*height)
            cv2.rectangle(img_rgb, pt1, pt2, (255,0,0), 2)

    msg = CvBridge().cv2_to_imgmsg(img_rgb, "bgr8")
    msg.header = header
    pub.publish(msg)


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
    print image_topic, model_def, weights_path, class_path, conf_thres, nms_thres, img_size
    #image_topic = '/camera/image_raw'
    
    
    # image_topic = '/camera/infra1/image_rect_raw'

    # prefix = rospkg.RosPack().get_path('yolov3_ros') + '/'
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_def", type=str, default=prefix+"config/yolov3-tiny-1class.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default=prefix+"weights/yolov3-tiny_drone.pth", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default=prefix+"config/drone.names", help="path to class label file")
    # parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    # parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    # global opt
    # opt = parser.parse_args()
    # # print(opt)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    rospy.Subscriber(image_topic, sensor_msgs.msg.Image, callback)

    # imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    # print("\nPerforming object detection:")
    # prev_time = time.time()
    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    #     # Configure input
    #     input_imgs = Variable(input_imgs.type(Tensor))

    #     # Get detections
    #     with torch.no_grad():
    #         detections = model(input_imgs)
    #         detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    #     # Log progress
    #     current_time = time.time()
    #     inference_time = datetime.timedelta(seconds=current_time - prev_time)
    #     prev_time = current_time
    #     print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    #     # Save image and detections
    #     imgs.extend(img_paths)
    #     img_detections.extend(detections)

    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")
    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    #     print("(%d) Image: '%s'" % (img_i, path))

    #     # Create plot
    #     img = np.array(Image.open(path))
    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)

    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
    #         detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
    #         unique_labels = detections[:, -1].cpu().unique()
    #         n_cls_preds = len(unique_labels)
    #         bbox_colors = random.sample(colors, n_cls_preds)
    #         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

    #             print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    #             box_w = x2 - x1
    #             box_h = y2 - y1

    #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #             # Create a Rectangle patch
    #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    #             # Add the bbox to the plot
    #             ax.add_patch(bbox)
    #             # Add label
    #             plt.text(
    #                 x1,
    #                 y1,
    #                 s=classes[int(cls_pred)],
    #                 color="white",
    #                 verticalalignment="top",
    #                 bbox={"color": color, "pad": 0},
    #             )

    #     # Save generated image with detections
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #     filename = path.split("/")[-1].split(".")[0]
    #     #plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    #     plt.savefig("output/{}.jpg".format(filename), bbox_inches="tight", pad_inches=0.0) 
    #     plt.close()
    
    rospy.spin()
