#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:48:01 2020

@author: sourabh
"""

import cv2
import numpy as np
import sift
import opticalflow
import multiview
import os

class Odometry():
    def __init__(self, image_prior, camera_matrix):
        self.sift  = sift.SIFT()
        self.tracker = opticalflow.Track()
        self.mv = multiview.MultiView()
        self.image_prior = image_prior        
        self.K = camera_matrix
        self.rotation = np.eye(3)
        self.translation = np.zeros((3,1))   
        _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True) 
        self.translation_history = np.array([0,0,0]).reshape(1,3)
        
    def mono_2d(self,image_current,heuristic_odom=True):
        if len(self.coords_prior) < 2000: ## Reinitialize kpts if tracked kpts falldown the threshold
            _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True)
        tracked_prior, tracked_current = self.tracker.KLT(self.image_prior, image_current, self.coords_prior)
        self.coords_prior = tracked_prior
        E, mask = cv2.findEssentialMat(tracked_prior, tracked_current, self.K, cv2.RANSAC,0.999,1.0); 
        _, R, t, mask = cv2.recoverPose(E, tracked_prior, tracked_current, self.K);         
        self.image_prior = image_current        
        if abs(t[1]) > 0.05 or abs(t[0]) > abs(0.8*t[2]) or not heuristic_odom: ## Conditions: Movement in vertical direction (assuming flat roads for now) is not possible, dominant direction of movement should always be forward
            return
        self.rotation = np.dot(R,self.rotation)
        self.translation = self.translation + np.dot(self.rotation,t)
        self.translation_history = np.vstack([self.translation_history, self.translation.reshape(1,3)])
        
    def draw_traj_2d(self,path="./",thickness=3):
        trajectory = (self.translation_history[:,0]*3).reshape(-1,1)
        trajectory = np.hstack([trajectory, (self.translation_history[:,2]*3).reshape(-1,1)])
        trajectory = (np.int_(trajectory))//3
        min_horiz = min(trajectory[:,0])
        max_horiz = max(trajectory[:,0])
        min_vertical = min(trajectory[:,1])
        max_vertical = max(trajectory[:,1])
        trajectory_plot = np.zeros((abs(min_horiz)+abs(max_horiz)+50,abs(min_vertical)+abs(max_vertical)+50,3))
        trajectory[:,0] += abs(min_horiz)+25
        trajectory[:,1] += abs(min_vertical)+25
        trajectory_plot[trajectory[:,1],trajectory[:,0]] = 255
        kernel = np.ones((thickness,thickness),np.uint8)
        dilation = cv2.dilate(trajectory_plot,kernel,iterations = 1)
        cv2.circle(trajectory_plot, (trajectory[0][0], trajectory[0][1]) ,1, (0,255,0), thickness*2);     
        cv2.circle(trajectory_plot, (trajectory[-1][0], trajectory[-1][1]) ,1, (0,0,255), thickness*2)
        cv2.imwrite(os.path.join(path,"trajectory.jpg"), trajectory_plot)
     
        
        
        