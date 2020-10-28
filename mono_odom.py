#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:31:05 2020

@author: sourabh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sift
import opticalflow
import pcd
import multiview
import glob
import os
import odometry
import visualize

sift  = sift.SIFT()
tracker = opticalflow.Track()
mv = multiview.MultiView()
viz = visualize.Visualize()
pcd = pcd.PCD()

path = "./data/data"
K = np.loadtxt("./camera_matrix.txt")

frame_old = cv2.imread(os.path.join(path, "000069.png"))
odom = odometry.Odometry(frame_old, K, track=True)

# for x in range(1,len(glob.glob(os.path.join(path,"*png")))):
for x in range(70,240):
    print(x)
    frame_new = cv2.imread(os.path.join(path, str(x).zfill(6)+".png"))
    odom.monocular_odometry(frame_new)    
    
    ## Uncomment for single instances
    # pointcloud = odom.get_monocular_pointcloud(frame_new)       
    # odom.forward(frame_new)
    # pointcloud = odom.pointcloud_history
    # trajectory = odom.pointcloud_trajectory_history
    # pointcloud = viz.pointcloud_trajectory(trajectory, pointcloud)    
    
    ## Only for Visualization
    # rot_x, rot_y = 950, 1200
    # trans_x, trans_y = 0, 0
    # scale = -15
    # save_path = os.path.join("./pointclouds", "overview"+str(x).zfill(6)+".jpg")
    # viz.save_pointcloud(pointcloud, save_path, rot_x, rot_y, trans_x, trans_y, scale, save=True)
    # viz.birds_eye_view(odom.trajectory_history,show=True)  
    
    
trajectory = viz.pointcloud_trajectory(odom.pointcloud_trajectory_history)
pcd.viz([odom.pointcloud_history]+trajectory, axis=True)

    
