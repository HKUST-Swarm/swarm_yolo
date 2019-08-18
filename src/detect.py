#! /usr/bin/env python

from __future__ import division

import sensor_msgs
import rospy
import cv2
from cv_bridge import CvBridge
from reproject import *

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import rospkg
import random

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from swarm_msgs.msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

global debug_show
debug_show = False
global count
count = 0

class BBox():
    def __init__(self, cx, cy, w, h, _id):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.id = _id
        
    def overlap(self, _bbox):
        XA1 = self.cx - self.w/2
        XA2 = self.cx + self.w/2
        XB1 = _bbox.cx - _bbox.w/2
        XB2 = _bbox.cx + _bbox.w/2

        YA1 = self.cy - self.h/2
        YA2 = self.cy + self.h/2
        YB1 = _bbox.cy - _bbox.h/2
        YB2 = _bbox.cy + _bbox.h/2
        
        SA = self.w*self.h
        SB = _bbox.w * _bbox.h
        
        SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
        SU = SA + SB - SI
        return SI / min(SA, SB)

class SwarmDetector:
    def __init__(self, img_size=416, conf_thres=0.9, nms_thres=0.1,debug_show="", weights_path="./weights/yolov3-tiny_drone.pth", model_def="config/yolov3-tiny-1class.cfg", class_path="config/drone.names"):
        self.bridge = CvBridge()
        self.debug_show = debug_show
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.history_bbox = []
        self.all_ids = set()
        self.MAX_XY_MATCHERR = 0.2
        self.MAX_Z_MATCHERR = 0.4
        self.MAX_DRONE_ID = 10

        self.detected_poses_pub = {}

        self.count = 0

        self.estimate_node_poses = {}

        self.pos = np.array([0, 0, 0])
        self.quat = np.array([0, 0, 0, 1])

        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Set up model
        self.model = model = Darknet(model_def, img_size=img_size).to(device)
        self.BBOX_TRACKS_FRAME = 3
        self.TRACK_OVERLAP_THRES = 0.2
        
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
        self.intrinsic = {
            "fx":384.12,
            "fy":384.12,
            "cx":319.95,
            "cy":234.29
        }
        
    def add_bbox(self, ts, bbox):
        if len(self.history_bbox) == 0 or self.history_bbox[-1]["ts"] != ts:
            #print("Add new bbox ts", ts, bbox)
            self.history_bbox.append(
                    {
                        "ts":ts,
                        "bboxes" : [bbox]
                    }
            )
        else:
            self.history_bbox[-1]["bboxes"].append(bbox)
    
    def update_pose(self, pos, quat):
        self.pos = pos
        self.quat = quat
        
    def update_estimate_poses(self, poses):
        self.estimate_node_poses = poses
    
    def project_estimate_node_img(self, _img):
        if self.pos is None:
            return
        #print (self.estimate_node_poses)
        for _id in self.estimate_node_poses:
            pose = self.estimate_node_poses[_id]
            target_pos = pose[0:3]
            target_quat = quaternion_from_euler(0, 0, pose[3])
            #print(target_pos, self.pos)
            draw_to_cv_image_(_img, target_pos, target_quat, self.pos, self.quat, "EST {}".format(_id))
    
    def match_pos(self, est_pos):
        for _id in self.estimate_node_poses:
            pose = self.estimate_node_poses[_id]
            target_pos = pose[0:3]
            err = est_pos - target_pos
            if np.abs(err[0]) < self.MAX_XY_MATCHERR and np.abs(err[1]) < self.MAX_XY_MATCHERR and np.abs(err[2]) < self.MAX_Z_MATCHERR:
                return _id
        return None

    def bbox_tracking(self, ts, cx, cy, w, h, est_pos):
        #print(est_pos)
        bbox = BBox(cx, cy, w, h, -1)

        _match_id = self.match_pos(est_pos)

        if len(self.history_bbox) > 0:
            min_index = len(self.history_bbox) - self.BBOX_TRACKS_FRAME
            if min_index < 0:
                min_index = 0
            for k in range(len(self.history_bbox)-1, min_index, -1):
                for _old_bbox in self.history_bbox[k]["bboxes"]:
                    #print("Overlap", _old_bbox.overlap(bbox))
                    if _old_bbox.overlap(bbox) > self.TRACK_OVERLAP_THRES:
                        #print("Track bounding box!")
                        if _old_bbox.id != _match_id:
                            if bbox.id > self.MAX_DRONE_ID:
                                bbox.id = _match_id #old bbox and this all should be matchid
                            else:
                                # We will accept old matched id
                                bbox.id = _old_bbox.id
                        else:
                            bbox.id = _old_bbox.id

                        self.add_bbox(ts, bbox)
                        return bbox.id  
        
        if _match_id is None:
            _id = random.randint(100,10000)
        else:
            _id = _match_id
        bbox.id = _id
        self.add_bbox(ts, bbox)
        return _id
    
    def predict_3dpose(self, cx, cy, d):
        tarpos_cam = np.array(XYZ_from_cxy_d(cx, cy, d, self.intrinsic)) #target position in camera frame
        r2 = quaternion_matrix(self.quat)[0:3,0:3]
        r2cam = np.dot(r2, R_cam_on_drone)
        dposyolo_global = np.dot(r2cam, tarpos_cam) + self.pos
        return dposyolo_global
    
    def pub_detection(self, _id, pos, stamp):
        if not(_id in self.detected_poses_pub):
            self.detected_poses_pub[_id] = rospy.Publisher("/swarm_detection/detected_pose", PoseStamped, queue_size=1)
        
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = stamp

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]

        self.detected_poses_pub[_id].publish(pose)

    def img_callback(self, img, depth_img):
        # print("GRAY", img.header.stamp)
        self.count += 1
        if self.count % 3 != 1:
            return
        ts = rospy.get_time()
        stamp = img.header.stamp
        img_gray = self.bridge.imgmsg_to_cv2(img, "mono8")
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        depth = CvBridge().imgmsg_to_cv2(depth_img, "32FC1") / 1000.0
        height, width = img_gray.shape[:2]

        loader = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_size = self.img_size
        img_ = cv2.resize(img_gray, (img_size, img_size))

        image = loader(img_).cuda()
        #image = image.view(1, img_size, img_size).expand(3, -1, -1)

        
        with torch.no_grad():
            detections = self.model(image.unsqueeze(0))
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        
        rospy.loginfo_throttle(1.0, "Detection use time {:f}ms".format((rospy.get_time() - ts)*1000))
        bboxes = []
        bboxes_unit = []
        if type(detections[0])==torch.Tensor:
            data = detections[0].numpy()
            for i in range(len(data)):
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0].numpy()[i,0:7]
                if x2 - x1 < 0.03*img_size or  y2 - y1 < 0.02*img_size:
                    continue

                pt1 = int(x1/img_size*width), int(y1/img_size*height)
                pt2 = int(x2/img_size*width), int(y2/img_size*height)
                
                bbox_width = (x2 - x1)/img_size*width
                cx = (x1+x2)/(img_size*2)
                cy = (y1+y2)/(img_size*2)
                w = (x2 - x1)/img_size
                h = (y2 - y1)/img_size
                
                d = estmate_distance(depth, cx, cy, self.intrinsic)
                
                xremin, yremin, xremax, yremax = reprojectBoundBox(cx, cy, d, self.intrinsic)
                
                width_re = xremax - xremin
                
                bbox_rate = width_re / bbox_width
                bbox_wrong = bbox_rate < 0.4 or bbox_rate > 1.6
                
              
                if self.debug_show != "":
                    if bbox_wrong:
                        cv2.rectangle(img_gray, pt1, pt2, (0, 0, 255), 2)
                        cv2.putText(img_gray, "WRONG BBOX RATE {:2.1f} D {:4.3f} RE width{}".format(bbox_rate, d, width_re), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
  0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(img_gray, pt1, pt2, (0, 255, 0), 2)
                        cv2.putText(img_gray, "RIGHT BBOX", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 255, 255), 1, cv2.LINE_AA)
                        
                if not bbox_wrong:
                    tarpos = self.predict_3dpose(cx, cy, d)
                    _id = self.bbox_tracking(stamp, cx, cy, w, h, tarpos)

                    if _id < self.MAX_DRONE_ID:
                        self.pub_detection(_id, tarpos, stamp)

                    cv2.putText(img_gray, "ID {}".format(_id), pt1, cv2.FONT_HERSHEY_SIMPLEX,
  1, (230, 25, 155), 2, cv2.LINE_AA)
                    bboxes.append((pt1, pt2))
                    bboxes_unit.append((cx, cy, w, h))
                
                


        rospy.loginfo_throttle(1.0, "Total use time {:f}ms".format((rospy.get_time() - ts)*1000))
        ret = []
        if self.debug_show != "":
            self.project_estimate_node_img(img_gray)
    
        if self.debug_show == "inline":
            _img = cv2.resize(img_gray, (1280, 960))
            plt.figure(0, figsize=(10,10))
            plt.imshow(_img)
            plt.show()
            

        elif self.debug_show == "cv":
            _img = cv2.resize(img_gray, (1280, 960))
            cv2.imshow("YOLO", _img)
            cv2.waitKey(1)
            

        
        
        return None
    
class SwarmDetectorNode:
    def __init__(self):
        image_topic = rospy.get_param('~image_topic')
        model_def = rospy.get_param('~model_def')
        weights_path = rospy.get_param('~weights_path')
        class_path = rospy.get_param('~class_path')
        self.conf_thres = conf_thres = rospy.get_param('~conf_thres', 0.9) # Upper than some value
        self.nms_thres = nms_thres = rospy.get_param('~nms_thres', 0.1)
        img_size = self.img_size = rospy.get_param('~img_size', 416)
        self.debug_show = rospy.get_param('~debug_show', False)

        print(image_topic, model_def, weights_path, class_path, conf_thres, nms_thres, img_size)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Using device {}".format(device))

        self.bridge = CvBridge()

        self.sd = SwarmDetector(model_def=model_def, 
            class_path=class_path, 
            img_size=img_size, 
            weights_path=weights_path, 
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            debug_show="cv")

        rospy.Subscriber("/camera/infra1/image_rect_raw", sensor_msgs.msg.Image, self.gray_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", sensor_msgs.msg.Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/swarm_drones/swarm_drone_fused", swarm_fused, self.swarm_drone_fused_callback, queue_size=1)
        rospy.Subscriber("/vins_estimator/imu_propagate", Odometry, self.vo_callback, queue_size=1)
        self.depth_img = None
        self.gray_img = None

        self.depths = None
    
    def vo_callback(self, odom):
        pos = odom.pose.pose.position
        att = odom.pose.pose.orientation
        pos = np.array([pos.x, pos.y, pos.z])
        quat = np.array([att.x, att.y, att.z, att.w])
        self.sd.update_pose(pos, quat)

    def swarm_drone_fused_callback(self, msg):
        drone_pos = {}
        for i in range(len(msg.ids)):
            _id = msg.ids[i]
            pos = msg.local_drone_position[i]
            drone_pos[_id] = np.array([
                pos.x, pos.y, pos.z, msg.local_drone_yaw[i]
            ])
        self.sd.update_estimate_poses(drone_pos)
     
    def get_depth(self, bboxes):
        if self.depths is None:
            return
        
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

    def depth_callback(self, depth_img):
        # print("DEPTH", depth_img.header.stamp)
        self.depth_img = depth_img
        if self.gray_img is not None and self.depth_img.header.stamp == self.gray_img.header.stamp:
            pass
            #print("Processing ", self.gray_img.header.stamp)
            #self.sd.img_callback(self.gray_img, self.depth_img)
        # self.depths = cv_image_array = np.array(self.depth_img, dtype = np.dtype('f8'))
        # cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        # self.depthimg = cv_image_norm
        # cv2.imshow("Image from my node", self.depthimg)
        # cv2.waitKey(1)

    def gray_callback(self, img_msg):
        self.gray_img = img_msg
        if self.depth_img is not None and self.gray_img.header.stamp == self.gray_img.header.stamp:
            #print("Processing ", self.gray_img.header.stamp)
            self.sd.img_callback(img_msg, self.depth_img)


if __name__ == "__main__":
    rospy.init_node('swarm_yolo')
    sd = SwarmDetectorNode()
    plt.ion()
    rospy.spin()
