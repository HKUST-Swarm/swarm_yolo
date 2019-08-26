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
#from torch2trt import torch2trt


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
    
    def toCVBOX(self):
        XA1 = self.cx - self.w/2
        XA2 = self.cx + self.w/2
        YA1 = self.cy - self.h/2
        YA2 = self.cy + self.h/2
        return int(XA1*640),int(YA1*480), int(self.w*640), int(self.h*480)


class SwarmDetector:
    def __init__(self, img_size=416, conf_thres=0.6, nms_thres=0.1,debug_show="", weights_trt_path="./weights/", weights_path="./weights/", model_def="config/yolov3-tiny-1class.cfg", class_path="config/drone.names"):
        self.bridge = CvBridge()
        self.debug_show = debug_show
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.history_bbox = []
        self.all_ids = set()
        self.MAX_XY_MATCHERR = 0.3
        self.MAX_Z_MATCHERR = 0.4
        self.MAX_DRONE_ID = 10

        self.detected_poses_pub = {}
        self.sw_detected_pub = rospy.Publisher("/swarm_detection/swarm_detected", swarm_detected, queue_size=1)

        self.count = 0

        self.estimate_node_poses = {}

        self.pos = np.array([0, 0, 0])
        self.quat = np.array([0, 0, 0, 1])

        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.camera_pos = np.array([0.044, -0.035, 0.0])

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.intrinsic = {
            "fx":384.12,
            "fy":384.12,
            "cx":319.95,
            "cy":234.29
        }

        self.multi_trackers = {}
        self.debug_tracker = None
        self.last_gray = None
        self.trackers_bboxs = {}
        self.WIDTH = 640.0
        self.HEIGHT = 480.0

        self.use_tensorrt = False
        self.first_init = True
        self.tracker_only_on_matched = True
        self.load_model(weights_path, weights_trt_path, model_def)  

        print("Model loaded; waiting for data")      

    
    def load_model(self, weights_path, weights_trt_path, model_def):
        img_size = self.img_size
        device = self.device
          # # Set up model
        if self.use_tensorrt:
            print("Using TENSORRT; Load TensorRT model")
            self.model_backbone = model_backbone = DarknetBackbone(model_def, img_size=img_size).to(device)
            self.model_end = model_end = DarknetEnd(model_def, img_size=img_size).to(device)
        else:
            self.model = model = Darknet(model_def, img_size=img_size).to(device)
        
        # classes = load_classes(class_path)  # Extracts class labels from file
        self.BBOX_TRACKS_FRAME = 3
        self.TRACK_OVERLAP_THRES = 0.2
        example_data = torch.zeros((1, 3, img_size, img_size)).cuda()
        # Load checkpoint weights
        if torch.cuda.is_available():
            if self.use_tensorrt:
                print(weights_trt_path)
                model_backbone.load_state_dict(torch.load(weights_path))
                model_end.load_state_dict(torch.load(weights_path))

                
                print("Convering model to TensorRT")
                self.model_backbone = model_backbone = torch2trt(model_backbone, [example_data], fp16_mode=True)
                print("Converting done;")

            else:
                model.load_state_dict(torch.load(weights_path))        
        else:
            if self.use_tensorrt:
                model_backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
                model_end.load_state_dict(torch.load(weights_path, map_location='cpu'))
            else:
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        if self.use_tensorrt:
            model_backbone.eval()  # Set in evaluation mode
            model_end.eval()  # Set in evaluation mode
        else:
            model.eval()  # Set in evaluation mode
            _ = model(example_data)

    
    def start_tracker_tracking(self, _id, frame, bbox):
        if _id in self.multi_trackers:
            del self.multi_trackers[_id]
        
        _tracker = cv2.TrackerMOSSE_create()   
        #_tracker = cv2.TrackerKCF_create()    
        #_tracker = cv2.TrackerCSRT_create()    
        
        #print("INIT tracker", bbox.toCVBOX())
        _tracker.init(frame, bbox.toCVBOX())
        self.multi_trackers[_id] = _tracker
        
            

    def tracker_draw_tracking(self, frame, frame_color, depth_img, stamp):
        self.trackers_bboxs = {}
        ids_need_to_delete = []
        objs = []
        count = 0
        for _id in self.multi_trackers:
            count += 1
            _tracker = self.multi_trackers[_id]
            (success, box) = _tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                x1, x2, y1, y2 = x, x + w, y, y + h
                bbox = self.trackers_bboxs[_id] = BBox((x+w/2.0)/self.WIDTH, (y+h/2.0)/self.HEIGHT, 
                                                w/ self.WIDTH, h/self.HEIGHT, _id)
                
                if _id < self.MAX_DRONE_ID and 0.99 > bbox.cx > 0.01 and 0.99 > bbox.cy > 0.01 :
                    bbox_wrong, d = self.reproject_check_bbx(bbox, depth_img)
                    if not bbox_wrong:
                        tarpos, dpos = self.predict_3dpose(bbox.cx, bbox.cy, d)
                        self.pub_detection(_id, tarpos, stamp)
                        objs.append((_id, dpos))                    
                
                if self.debug_show != "":
                    cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 100, 100), 2)
                    cv2.putText(frame_color, "TRACKER {}".format(_id), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
            else:
                rospy.loginfo("TRACKER {} failed".format(_id))
                ids_need_to_delete.append(_id)
        rospy.loginfo_throttle(3.0, "Is tracking {} objects".format(count))
        for _id in ids_need_to_delete:
            del self.multi_trackers[_id]

        return objs
            
    def add_bbox(self, ts, bbox):
        if len(self.history_bbox) == 0 or self.history_bbox[-1]["ts"] != ts:
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
        _match_id = None
        _match_err_norm = 1000
        for _id in self.estimate_node_poses:
            pose = self.estimate_node_poses[_id]
            target_pos = pose[0:3]
            err = est_pos - target_pos
            #print("Matching EST {} with current {}".format(_id, err))
            err_norm = np.linalg.norm(err)
            if np.abs(err[0]) < self.MAX_XY_MATCHERR and np.abs(err[1]) < self.MAX_XY_MATCHERR and np.abs(err[2]) < self.MAX_Z_MATCHERR and err_norm < _match_err_norm:
                _match_id = _id
                _match_err_norm = err_norm
        if _match_id is not None:
            #print("Matched ", _match_id)
            pass
        return _match_id
    
    def estimate_bbox_id(self, bbox):
        for _id in self.trackers_bboxs:
            _tracked_bbox = self.trackers_bboxs[_id]
            if _tracked_bbox.overlap(bbox) > 0.6:
                return _id
        #return None
        if len(self.history_bbox) > 0:
            min_index = len(self.history_bbox) - self.BBOX_TRACKS_FRAME
            if min_index < 0:
                min_index = 0
            best_overlap =  self.TRACK_OVERLAP_THRES 
            best_id = None
            for k in range(len(self.history_bbox)-1, min_index, -1):
                for _old_bbox in self.history_bbox[k]["bboxes"]:
                    #print("Overlap", _old_bbox.overlap(bbox))
                    if _old_bbox.overlap(bbox) > best_overlap:
                        best_id =  _old_bbox.id
                        best_overlap = _old_bbox.overlap(bbox)
                if best_id is not None:
                    return best_id
            return best_id
                    
    def remove_tracker(self, _id):
        if _id in self.multi_trackers:
            del self.multi_trackers[_id]
        
    def bbox_tracking(self, ts, bbox, est_pos, frame_gray):

        #if self.debug_tracker is None:
        _match_id = self.match_pos(est_pos)
        _id = self.estimate_bbox_id(bbox)
        
        if _id is None:
            _id = random.randint(100, 1000)
            rospy.loginfo("Found new object {}".format(_id))
        else:
            self.remove_tracker(_id)
        #Start new track; fix the bounding box
            
        if _id > self.MAX_DRONE_ID and _match_id is not None:
            _id = _match_id
            rospy.loginfo("Object {} match to Drone {}".format(_id, _match_id))
        
        bbox.id = _id

        if (not self.tracker_only_on_matched) or (self.tracker_only_on_matched and _id < self.MAX_DRONE_ID):
            self.start_tracker_tracking(_id, frame_gray, bbox)
            
        self.add_bbox(ts, bbox)
        
        return _id
    
    def predict_3dpose(self, cx, cy, d):
        tarpos_cam = np.array(XYZ_from_cxy_d(cx, cy, d, self.intrinsic)) #target position in camera frame
        r2 = quaternion_matrix(self.quat)[0:3,0:3]
        
        dpos_body_frame = np.dot(R_cam_on_drone, tarpos_cam) + self.camera_pos
        dposyolo_global = np.dot(r2, dpos_body_frame)
        
        pos_global = dposyolo_global + self.pos
        
        return pos_global, dpos_body_frame
    
    def pub_detection(self, _id, pos, stamp):
        if not(_id in self.detected_poses_pub):
            self.detected_poses_pub[_id] = rospy.Publisher("/swarm_detection/detected_pose_{}".format(_id), PoseStamped, queue_size=1)
        
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = stamp

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]

        self.detected_poses_pub[_id].publish(pose)

    def publish_alldetected(self, detected_objects, stamp):
        sw_detected = swarm_detected()
        sw_detected.header.stamp = stamp
        sw_detected.self_drone_id = -1
        for _id, dpos in detected_objects:
            nd = node_detected()
            nd.header.stamp = stamp
            nd.self_drone_id = -1
            nd.remote_drone_id = _id
            nd.is_2d_detect = False
            nd.is_yaw_valid = False
            nd.relpose.pose.position.x = dpos[0]
            nd.relpose.pose.position.y = dpos[1]
            nd.relpose.pose.position.z = dpos[2]
            nd.relpose.covariance[0] = 0.1*0.1
            nd.relpose.covariance[6+1] = 0.07*0.07
            nd.relpose.covariance[2*6+2] = 0.07*0.07

            sw_detected.detected_nodes.append(nd)

        self.sw_detected_pub.publish(sw_detected)
    
    def detect_by_yolo(self, img_gray):
        ts = rospy.get_time()
        loader = transforms.Compose([
            transforms.ToTensor(),
        ])


        img_size = self.img_size
        img_ = cv2.resize(img_gray, (img_size, img_size))

        image = loader(img_).cuda()
        
        with torch.no_grad():
            if not self.use_tensorrt:
                detections = self.model(image.unsqueeze(0))
                #print("DTS ", detections.shape)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            else:
                #print("Using converted")
                x, x8 = self.model_backbone(image.unsqueeze(0))
                #print("Finish backbone", x.shape, x8.shape, x13.shape)
                detections = self.model_end(x, x8)
                #print("Finish end", detections.shape)
                #print(detections)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            
        rospy.loginfo_throttle(1.0, "Detection use time {:f}ms".format((rospy.get_time() - ts)*1000))
        if detections[0] is not None:
            return detections[0].cpu().numpy()
        return []

    def draw_bbox_debug_info(self, bbox_wrong, img_gray, pt1, pt2, bbox_rate, d, width_re):
        if self.debug_show != "":
            if bbox_wrong:
                        cv2.rectangle(img_gray, pt1, pt2, (0, 0, 255), 2)
                        cv2.putText(img_gray, "WRONG BBOX RATE {:2.1f} D {:4.3f} RE width{}".format(bbox_rate, d, width_re), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img_gray, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(img_gray, "RIGHT BBOX", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

    def show_debug_img(self, img_):
        if self.debug_show != "":
            self.project_estimate_node_img(img_)
    
        if self.debug_show == "inline":
            #_img = cv2.resize(img_gray, (1280, 960))
            plt.figure(0, figsize=(10,10))
            plt.imshow(img_)
            plt.show()
            

        elif self.debug_show == "cv":
            #_img = cv2.resize(img_gray, (1280, 960))
            cv2.imshow("YOLO", img_)
            cv2.waitKey(2)

    def reproject_check_bbx(self, bbx, depth, img_ = None):
        bbox_width = bbx.w * self.WIDTH
        # print(bbx.cx, bbx.cy)
        d = estimate_distance(depth, bbx.cx, bbx.cy, self.intrinsic)
        xremin, yremin, xremax, yremax = reprojectBoundBox(bbx.cx, bbx.cy, d, self.intrinsic)
        
        width_re = xremax - xremin
        
        bbox_rate = width_re / bbox_width
        #print("RE WITDH {} BBOX WIDTH {}".format(width_re, bbox_width))
        bbox_wrong = bbox_rate < 0.4 or bbox_rate > 1.6
        X1, Y1, W, H = bbx.toCVBOX()
        if img_ is not None:
            self.draw_bbox_debug_info(bbox_wrong, img_, (X1, Y1), (X1 + W, Y1 + H), bbox_rate, d, width_re)

        return bbox_wrong, d

    def img_callback(self, img, depth_img, force_no_disp=False):
        img_size = self.img_size
        ts = rospy.get_time()
        stamp = img.header.stamp
        img_gray = self.bridge.imgmsg_to_cv2(img, "mono8")
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        if self.debug_show is not None:
            img_to_draw = img_color.copy()
        depth = CvBridge().imgmsg_to_cv2(depth_img, "32FC1") / 1000.0
        height, width = img_gray.shape[:2]


        self.count += 1
        detected_objects = self.tracker_draw_tracking(img_gray, img_to_draw, depth, stamp)
        rospy.loginfo_throttle(1.0, "DT stamp {}".format(stamp - depth_img.header.stamp))
        if self.count % 3 != 1:
            rospy.loginfo_throttle(1.0, "Total use time {:f}ms".format((rospy.get_time() - ts)*1000))
            return img_to_draw

        data = self.detect_by_yolo(img_color)

        for i in range(len(data)):
            x1, y1, x2, y2, conf, cls_conf, cls_pred = data[i,0:7]
            if x2 - x1 < 0.03*img_size or  y2 - y1 < 0.02*img_size:
                continue
            
            bbox = BBox((x1+x2)/2.0/img_size, (y1+y2)/2.0/img_size, float(x2 - x1)/ img_size, float(y2 - y1)/img_size, None)
            
            bbox_wrong, d = self.reproject_check_bbx(bbox, depth, img_to_draw)

            if not bbox_wrong:
                tarpos, dpos = self.predict_3dpose(bbox.cx, bbox.cy, d)
                _id = self.bbox_tracking(stamp, bbox, tarpos, img_gray)
                #print("Not Wrong bbx")

                if _id < self.MAX_DRONE_ID:
                    self.pub_detection(_id, tarpos, stamp)
                    detected_objects.append((_id, dpos))
                    
                if self.debug_show =="cv":
                    x1, y1, _, _ = bbox.toCVBOX()
                    cv2.putText(img_to_draw, "ID {}".format(_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 25, 155), 2, cv2.LINE_AA)
            
                
        rospy.loginfo_throttle(1.0, "Total use time {:f}ms".format((rospy.get_time() - ts)*1000))
        if not force_no_disp:
            self.show_debug_img(img_to_draw)

        self.publish_alldetected(detected_objects, stamp)
        return img_to_draw
    
class SwarmDetectorNode:
    def __init__(self):
        image_topic = rospy.get_param('~image_topic')
        model_def = rospy.get_param('~model_def')
        weights_path = rospy.get_param('~weights_path')
        weights_trt_path = rospy.get_param('~weights_trt_path')
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
            weights_trt_path=weights_trt_path, 
            weights_path=weights_path, 
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            debug_show=self.debug_show)

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
        #print("DEPTH_CB", (depth_img.header.stamp.to_sec()*1000000)%1000000)
        self.depth_img = depth_img
        if self.gray_img is not None and self.depth_img.header.stamp == self.gray_img.header.stamp:
            self.sd.img_callback(self.gray_img, self.depth_img)
        
        # self.depths = cv_image_array = np.array(self.depth_img, dtype = np.dtype('f8'))
        # cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        # self.depthimg = cv_image_norm
        # cv2.imshow("Image from my node", self.depthimg)
        # cv2.waitKey(1)

    def gray_callback(self, img_msg):
        #print("GRAY_CB", (img_msg.header.stamp.to_sec()*1000000)%1000000)        
        self.gray_img = img_msg
        if (self.depth_img is not None) and (self.depth_img.header.stamp == self.gray_img.header.stamp):
            self.sd.img_callback(img_msg, self.depth_img, True)


if __name__ == "__main__":
    rospy.init_node('swarm_yolo')
    sd = SwarmDetectorNode()
    plt.ion()
    rospy.spin()
