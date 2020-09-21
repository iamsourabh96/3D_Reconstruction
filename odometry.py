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
    
    def __init__(self, image_prior, camera_matrix, track=False):
        '''
        Parameters
        ----------
        image_prior : Initialize the odomotery object with first image in the video sequence
        Odomotery can then be obtained by passing the subsequent frames to the monocular_odometry function. 
        camera_matrix : Intrinsic parameteres of camera
        track: set False to use SIFT features on all frames or set True to track features on subsequent frames.
        Tracking features approach is faster than computing SIFT features on all frames
        '''
        self.sift  = sift.SIFT()
        self.tracker = opticalflow.Track()
        self.mv = multiview.MultiView()
        self.pcd = pcd.PCD()
        self.image_prior = image_prior        
        self.K = camera_matrix
        self.track_features = track
        _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True) 
        self.rotation = np.eye(3)
        self.translation = np.zeros((3,1))
        self.trajectory_history = np.array([]).reshape(-1,3)
        self.pointcloud_history = np.array([]).reshape(-1,3)
        self.pointcloud_trajectory_history = np.zeros((1,3))
        self.flag = 0

        
    def forward(self, image_current):
        self.image_prior = image_current
        self.flag = 0
        
    
    def monocular_odometry(self, image_current, num_points=10000):
        '''
        Parameters
        ----------
        image_current : subsequent frames from video should be passed here
        num_points : number of pointclouds 
        Description: This is the main function for odometry and will handle all the steps needed. 
        For extraction of individual parameters try get functions.
        '''
        self.flag = 1
        self.monocular(image_current) ## Gets the universal rotation, translation and trajectory
        proj_mat1, proj_mat2 = self.get_monocular_projection_matrices(self.K, self.R, self.t)
        self.monocular_pointcloud(self.kpts_prior[:num_points], self.kpts_current[:num_points], proj_mat1, proj_mat2)
        self.pointcloud_trajectory(skip_frames=60,curve=0.11) ## Creates trajectory with minimal points (vertices) 
        self.forward(image_current)  ## Proceeds to next frame (Assuming next frame will be provided for next function call)

        
    def monocular(self,image_current, heuristic_odom=True):
        R, t = self.get_relative_orientation(image_current)        
        if abs(t[1]) > 0.05 or abs(t[0]) > abs(0.8*t[2]) or not heuristic_odom: ## Conditions: Movement in vertical direction (assuming flat roads for now) is not possible, dominant direction of movement should always be forward
            self.trajectory_history = np.vstack([self.trajectory_history, self.translation.reshape(1,3)])
        else:
            self.rotation = np.dot(R,self.rotation)
            self.translation = self.translation + (np.dot(self.rotation,t))
            self.trajectory_history = np.vstack([self.trajectory_history, self.translation.reshape(1,3)])

    
    def get_relative_orientation(self, image_current):        
        kpts_prior, kpts_current = self.get_keypoints(image_current)    
        E, mask = cv2.findEssentialMat(kpts_prior, kpts_current, self.K, cv2.RANSAC,0.999,1.0); 
        _, R, t, mask = cv2.recoverPose(E, kpts_prior, kpts_current, self.K); 
        if self.flag:
            self.R, self.t = R, t
        return R, t
    
    
    def get_keypoints(self,image_current):
        if self.track_features:
            if len(self.coords_prior) < 1500 : ## Reinitialize kpts if tracked kpts fall below the threshold
                _, _, self.coords_prior = self.sift.features(self.image_prior, coord=True)        
            kpts_prior, kpts_current = self.tracker.KLT(self.image_prior, image_current, self.coords_prior) ## Using optical flow to track keypoints to next frame
        else:
            kpts_prior, kpts_current = self.sift.find_correspondences(self.image_prior, image_current)      
        if self.flag:
            self.kpts_prior, self.kpts_current = kpts_prior, kpts_current
            self.coords_prior = kpts_current
        return kpts_prior, kpts_current
    
    
    def get_monocular_projection_matrices(self,K,R,t):
        return self.mv.get_proj_matrix(K), self.mv.get_proj_matrix(K,R,t)
    
    
    def get_monocular_pointcloud(self, image_current, num_points=1000): ## Gets monocular pointcloud provided an image
        points1, points2 = self.get_keypoints(image_current)
        R, t = self.get_relative_orientation(image_current)
        proj_mat1, proj_mat2 = self.get_monocular_projection_matrices(self.K,R,t)        
        pointcloud = self.mv.triangulate(points1[:num_points],points2[:num_points],proj_mat1,proj_mat2)
        pointcloud = self.pcd.filtr(pointcloud, x_limit=(-25,25),y_limit=(-10,5),z_limit=(0,40), inclusive=True) ## Include only nearby values in pointcloud
        return pointcloud
   
    
    def monocular_pointcloud(self, points1, points2, proj_mat1, proj_mat2):
        pointcloud = self.mv.triangulate(points1,points2,proj_mat1,proj_mat2)
        pointcloud = self.pcd.filtr(pointcloud, x_limit=(-25,25),y_limit=(-50,10),z_limit=(0,15), inclusive=True) ## Include only nearby values in pointcloud
        translation = self.translation*(np.array([-1,1,-1]).reshape(3,1))
        self.pointcloud = self.mv.transform(pointcloud, np.hstack([self.rotation, translation]))  ## Transform pointcloud to the current trajectory of the camera location
        self.pointcloud_history = np.vstack([self.pointcloud_history, self.pointcloud])
        

    def pointcloud_trajectory(self,skip_frames=5, curve=0.25): ## Workaround to get LineMesh object running (Cannot display large number of LineMesh objects)
        translation = self.translation*(np.array([-1,1,-1]).reshape(3,1)) 
        if abs(self.t[0])>curve: ## Adds a vertex if deviation in the horizontal axis is detected (Curve values should be between 0 and 1)
            self.pointcloud_trajectory_history = np.vstack([self.pointcloud_trajectory_history, translation.T])
        elif not len(self.trajectory_history)%skip_frames:  ## Add a vertex after 'n' frames (specified by skip frames parameter)
            self.pointcloud_trajectory_history = np.vstack([self.pointcloud_trajectory_history, translation.T])
