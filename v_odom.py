#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:36:28 2020

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

sift  = sift.SIFT()
tracker = opticalflow.Track()
mv = multiview.MultiView()

path = "./data/data"
K = np.loadtxt("./camera_matrix.txt")
traj = np.zeros((800,1000,3),dtype=np.uint8)

rot_universal = np.eye(3)
trans_universal = np.zeros((3,1))

frame_old = cv2.imread(os.path.join(path, "000000.png"))
# for x in range(1,len(glob.glob(os.path.join(path,"*png")))):
for x in range(70,250):
    print(x)
    kp_old, des_old, coords_old = sift.features(frame_old, coord=True)
    frame_new = cv2.imread(os.path.join(path, str(x).zfill(6)+".png"))
    tracked_old, tracked_new = tracker.KLT(frame_old, frame_new, coords_old)

    E, mask = cv2.findEssentialMat(tracked_old, tracked_new, K, cv2.RANSAC,0.999,1.0); 
    _, R, t, mask = cv2.recoverPose(E, tracked_old, tracked_new, K);      

    frame_old = frame_new

    if abs(t[1]) > 0.05 or abs(t[0]) > abs(0.8*t[2]):
        continue
    
    rot_universal = np.dot(R,rot_universal)
    trans_universal = trans_universal + np.dot(rot_universal,t)
        
    draw_x, draw_y = int(trans_universal[0])+500, int(trans_universal[2])+500
    cv2.circle(traj, (draw_x, draw_y) ,1, (255,255,255), 2)
    
    cv2.imshow("current frame", frame_new)
    cv2.imshow("trajectory", traj)
    cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2);     
    
    if cv2.waitKey(1) & 0xFF == 27:
        break 
cv2.destroyAllWindows()
cv2.imwrite("test.jpg", traj) 



