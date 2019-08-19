#!/usr/bin/env python

import rosbag
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from tf.transformations import *
from IPython import display
import os
def read_bag(_bag):
    poses2index = {
        "/swarm_mocap/SwarmNodePose0":0,
        "/swarm_mocap/SwarmNodePose2":2,
        "/swarm_mocap/SwarmNodePose4":4,
    }
    camera = []
    depth = []
    
    poses = {
        0:[],
        2:[],
        4:[]
    }
    for topic, msg, t in _bag.read_messages(topics=["/swarm_mocap/SwarmNodePose0", "/swarm_mocap/SwarmNodePose2", "/swarm_mocap/SwarmNodePose4"]):
        poses[poses2index[topic]].append(msg)

    for topic, msg, t in _bag.read_messages(topics=["/camera/infra1/image_rect_raw", "/camera/depth/image_rect_raw"]):
        if topic == "/camera/infra1/image_rect_raw":
            camera.append(msg)
        if topic == "/camera/depth/image_rect_raw":
            depth.append(msg)
        
    return poses, camera, depth

def parse_ros_pose_ts(poses):
    d = {
        "ts":[],
        "pos":[],
        "quat":[]
    }
    for msg in poses:
        pos = msg.pose.position
        att = msg.pose.orientation
        d["ts"].append(msg.header.stamp.to_sec())
        d["pos"].append([pos.x, pos.y, pos.z])
        d["quat"].append([att.x, att.y, att.z, att.w])
    d["pos"] = np.array(d["pos"])
    d["quat"] = np.array(d["quat"])
    d["ts"] = np.array(d["ts"])
    d["pos_func"] = interp1d(d["ts"], d["pos"], axis=0)
    d["quat_func"] = interp1d(d["ts"], d["quat"], axis=0)
    return d
K_cam_gray = np.array([[384.12371826171875, 0.0, 319.9548645019531],
                  [0.0, 384.12371826171875, 234.2923126220703], 
                  [0.0, 0.0, 1.0]])
K_cam_color = np.array([1375.4482421875, 0.0, 950.56201171875, 0.0, 1375.8404541015625, 566.9019165039062, 0.0, 0.0, 1.0])
K_cam_color = K_cam_color.reshape((3,3))
print(K_cam_color)

D_cam = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
R_cam_on_drone = np.array([[0, 0, 1],
         [-1, 0, 0],
         [0, -1, 0]])
t_cam_on_drone = np.array([0.044, -0.035, 0.0])
plt.ion()

def draw_image(img, points, xmin, xmax, ymin, ymax):
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    for point in points:
        cv2.circle(img, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0) , 1)

def draw_to_cv_image_(_img, d0_pos, d0_quat, d2_pos, d2_quat, text):
    width = 640
    height = 320
    d0_fix = np.array([0, 0, 0])
    r = quaternion_matrix(d2_quat)[0:3,0:3]
    R0 = quaternion_matrix(d0_quat)[0:3, 0:3]
    rvec = np.transpose(np.dot(r, R_cam_on_drone))
    tvec = -np.dot(rvec, d2_pos)
    
    corpoints = np.array([
         d0_pos + np.dot(R0, np.array([0.145, 0.145, -0.02]) + d0_fix), #Center
         d0_pos + np.dot(R0, np.array([0.145, 0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([0.145, -0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([-0.145, 0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([-0.145, -0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([0, 0, 0.13]) + d0_fix),
     ])
    


    points, _ = cv2.projectPoints(corpoints, rvec, tvec, K_cam_gray, D_cam)
    #_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    xmin = int(np.min(points[:,:,0]))
    xmax = int(np.max(points[:,:,0]))
    ymin = int(np.min(points[:,:,1]))
    ymax = int(np.max(points[:,:,1]))
    
    draw_image(_img, points, xmin, xmax, ymin, ymax)
    cv2.putText(_img, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 155), 3, cv2.LINE_AA)
    c_x = (xmin + xmax)/(2*width)
    c_y = (ymin + ymax)/(2*height)
    
    w = (xmax - xmin)/(width)
    h = (ymax - ymin)/(height)
    return c_x, c_y, w, h, _img

    
def project_to_cv_image(img_msg, dp2, dp0, d0_fix, display = "cv", img_to_draw = None):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, "mono8")
    try:
        d0_pos = dp0["pos_func"](img_msg.header.stamp.to_sec())
        d0_quat = dp0["quat_func"](img_msg.header.stamp.to_sec())
        d2_quat = dp2["quat_func"](img_msg.header.stamp.to_sec())
        d2_pos = dp2["pos_func"](img_msg.header.stamp.to_sec()) + t_cam_on_drone
    except:
        return  None,None,None,None,None,None, cv_image, cv_image
    d2_quat = quaternion_multiply(d2_yaw_fix, d2_quat)
    r = quaternion_matrix(d2_quat)[0:3,0:3]
    R0 = quaternion_matrix(d0_quat)[0:3, 0:3]
    rvec = np.transpose(np.dot(r, R_cam_on_drone))
    tvec = - np.dot(rvec, d2_pos)
    
    rpy2 = euler_from_quaternion(d2_quat)
    rpy0 = euler_from_quaternion(d0_quat)
    dis = np.linalg.norm(d2_pos - d0_pos - d0_fix)
#     print("Distance ", )
    
    corpoints = np.array([
         d0_pos + np.dot(R0, np.array([0.145, 0.145, -0.02]) + d0_fix), #Center
         d0_pos + np.dot(R0, np.array([0.145, 0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([0.145, -0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([-0.145, 0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([-0.145, -0.145, -0.02]) + d0_fix),
         d0_pos + np.dot(R0, np.array([0, 0, 0.13]) + d0_fix),
     ])
    


    points, _ = cv2.projectPoints(corpoints, rvec, tvec, K_cam_gray, D_cam)
    _img = cv_image
    #_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    xmin = int(np.min(points[:,:,0]))
    xmax = int(np.max(points[:,:,0]))
    ymin = int(np.min(points[:,:,1]))
    ymax = int(np.max(points[:,:,1]))
    
#     print(xmin, xmax)
#     _img = cv2.resize(_img, (1280, 960), interpolation=cv2.INTER_CUBIC)
    if ymax > img_msg.height or ymin < 0 or xmax > img_msg.width or xmin < 0:
        return None,None,None,None,None,None, cv_image, cv_image

    if display=="cv":
        if img_to_draw is None:
            img_to_draw = _img.copy()
        draw_image(img_to_draw, points, xmin, xmax, ymin, ymax)
        img_to_show = cv2.resize(img_to_draw, (1280, 960))
        cv2.imshow("img", img_to_show)
        cv2.waitKey(30)
    elif display == "inline":
        if img_to_draw is None:
            img_to_draw = _img.copy()
        draw_image(img_to_draw, points, xmin, xmax, ymin, ymax)
        img_to_draw = cv2.resize(img_to_draw, (1280, 960))
        plt.figure()
        plt.imshow(img_to_draw)
#         plt.pause(0.1)
    elif display=="draw_only":
        if img_to_draw is None:
            img_to_draw = _img.copy()
        draw_image(img_to_draw, points, xmin, xmax, ymin, ymax)
        
    xmin = np.min(points[:,:,0])
    xmax = np.max(points[:,:,0])
    ymin = np.min(points[:,:,1])
    ymax = np.max(points[:,:,1])
    c_x = (xmin + xmax)/(2*img_msg.width)
    c_y = (ymin + ymax)/(2*img_msg.height)
    
    w = (xmax - xmin)/(img_msg.width)
    h = (ymax - ymin)/(img_msg.height)
    dyaw = rpy0[2] - rpy2[2]
    return c_x, c_y, w, h, dyaw, dis, _img, img_to_draw

    
def crop_depth(depth, c_x, c_y, w, h):
    x0 = int((c_x - w/2)*640)
    x1 = int((c_x + w/2)*640)

    y0 = int((c_y - h/2)*480)
    y1 = int((c_y + h/2)*480)
    crop_dp1 = depth[y0:y1, x0:x1]
    return crop_dp1

def XYZ_from_cxy_d(cx, cy, d, intrinsic):
    u = cx*640
    v = cy*480

    X = (u - intrinsic["cx"]) * d / intrinsic["fx"]
    Y = (v - intrinsic["cy"]) * d / intrinsic["fy"]
    return X, Y, d

def distance_from_cxy_d(cx, cy, d, intrinsic):
    x, y, z = XYZ_from_cxy_d(cx, cy, d, intrinsic)
    return math.sqrt(x**2 + y**2 + z**2)  + 0.15

def reprojectBoundBox(cx, cy, d, intrinsic):
    #print(cx, cy)
    x, y, z = XYZ_from_cxy_d(cx, cy, d, intrinsic)
    #print("XYZ on cam", x, y, z)
    d0_pos = np.array([x, y, z])

    corpoints = np.array([
         d0_pos + np.array([0.145,  -0.02, 0.145]), #Center
         d0_pos + np.array([0.145, -0.02, 0.145]),
         d0_pos + np.array([0.145, -0.02, -0.145]),
         d0_pos + np.array([-0.145, -0.02, 0.145]),
         d0_pos + np.array([-0.145, -0.02, -0.145]),
         d0_pos + np.array([0, 0.13, 0]),
     ])

    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 0.0])
    points, _ = cv2.projectPoints(corpoints, rvec, tvec, K_cam_gray, D_cam)

    #_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    xmin = int(np.min(points[:,:,0]))
    xmax = int(np.max(points[:,:,0]))
    ymin = int(np.min(points[:,:,1]))
    ymax = int(np.max(points[:,:,1]))

    return int(xmin), int(ymin), int(xmax), int(ymax)

def estimate_distance(depth, c_x, c_y, intrinsic, avr_range = 5):
    width = 640.0
    height = 480.0
    c_x = int(c_x*width)
    c_y = int(c_y*height)
    d1 = depth[c_y, c_x]/1000.0
    if d1 > 0.01:
        return distance_from_cxy_d(c_x, c_y, d1, intrinsic)

    ymin = c_y - avr_range
    if ymin < 0:
        ymin = 0

    ymax = c_y + avr_range
    if ymin > height:
        ymin = height

    xmin = c_x - avr_range
    if xmin < 0:
        xmin = 0

    xmax = c_x + avr_range
    if xmax > width:
        xmax = width


    dis_count = 0
    dissum = 0
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            if depth[i][j] > 0.01:
                dis_count += 1
                dissum += distance_from_cxy_d(j/width, i/height, depth[i][j], intrinsic)

    if dis_count == 0:
        print("Dis is 0", c_x, c_y)
        return 0
    return dissum / dis_count
