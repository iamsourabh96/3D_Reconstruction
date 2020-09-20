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

frame_old = cv2.imread(os.path.join(path, "000000.png"))
odom = odometry.Odometry(frame_old, K, track=True)

# for x in range(1,len(glob.glob(os.path.join(path,"*png")))):
for x in range(70,240):
    print(x)
    frame_new = cv2.imread(os.path.join(path, str(x).zfill(6)+".png"))
    odom.monocular_odometry(frame_new)        
    viz.birds_eye_view(odom.trajectory_history,show=True)    
    viz.pointcloud_trajectory(trajectory=odom.pointcloud_trajectory_history, pointcloud=[odom.pointcloud_history])

    