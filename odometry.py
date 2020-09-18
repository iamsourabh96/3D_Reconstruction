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
import pcd

class Odometry():
    
    def __init__(self, image_prior, camera_matrix):
        self.sift  = sift.SIFT()
        self.tracker = opticalflow.Track()
        self.mv = multiview.MultiView()
        self.pcd = pcd.PCD()
        self.image_prior = image_prior        
        self.K = camera_matrix
        _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True) 
        self.rotation = np.eye(3)
        self.translation = np.zeros((3,1))
        self.trajectory_history = np.array([0,0,0]).reshape(1,3)
        
        
    def monocular(self,image_current,sparse=True, heuristic_odom=True):
        if len(self.coords_prior) < 2000 or not sparse: ## Reinitialize kpts if tracked kpts falldown the threshold
            _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True)
        tracked_prior, tracked_current = self.tracker.KLT(self.image_prior, image_current, self.coords_prior)
        self.coords_prior = tracked_prior
        E, mask = cv2.findEssentialMat(tracked_prior, tracked_current, self.K, cv2.RANSAC,0.999,1.0); 
        _, R, t, mask = cv2.recoverPose(E, tracked_prior, tracked_current, self.K);         
        self.image_prior = image_current        
        if abs(t[1]) > 0.05 or abs(t[0]) > abs(0.8*t[2]) or not heuristic_odom: ## Conditions: Movement in vertical direction (assuming flat roads for now) is not possible, dominant direction of movement should always be forward
            return  tracked_prior, tracked_current, R, t
        self.rotation = np.dot(R,self.rotation)
        self.translation = self.translation + np.dot(self.rotation,t)
        self.trajectory_history = np.vstack([self.trajectory_history, self.translation.reshape(1,3)])
        return  tracked_prior, tracked_current, R, t
    
    def mono_pointcloud(self,points1,points2,K,R,t):
        P1 = self.mv.get_proj_matrix(K)
        P2 = self.mv.get_proj_matrix(K,R,t)
        pointcloud = (self.mv.triangulate(points1,points2,P1,P2))
        # (success, rvec, tvec) = cv2.solvePnP(pointcloud, points1, K, distCoeffs=0, flags=cv2.cv2.SOLVEPNP_ITERATIVE)        
        pointcloud = self.pcd.filtr(pointcloud)
        return pointcloud        
        