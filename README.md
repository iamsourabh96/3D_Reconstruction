# Monocular Visual Odometry
A simple monocular visual odometry implementation with 3 components: 
1. Feature Detection.
2. Computing Essential Matrix.
3. Estimating Relative Orientation.


## Feature Detection and Tracking

My approach uses SIFT feature detectors but this can be easily swapped with other feature detectors like SURF, ORB or FAST corner detector.
The current implementation has two approaches to feature tracking:   
a. Detecting features on every subsequent frames and obtaining correspondences.  
b. Using Optical flow to track features to subsequent frames.

The tracking features approach is computationally faster. Tracked features can be lost over the subsequent frames and will have to be reintiallized once they fall below a certain specified threshold. In my approach the features are reinitialized if the number of key points fall below 1500.


## Computing Essesntial Matrix from point correspondences

Given the camera intrinsic matrix, Essential Matrix can then be computed from point correspondences using the 5-point algorithm. 
Five points would be sufficient if the point correspondences were perfect but since this is not the case, RANSAC is used to filter outliers.


## Estimating the Relative Orientation from essential matrix

The essential matrix is then decomposed into a 3x3 rotation matrix and a 3x1 translation vector. Assuming the camera pose for frame0 is at origin (Identity rotation and zero translation), the relative orientation of the camera pose in frame1 wrt frame0 is obtained from the essential matrix.  
The essential matrix has only 5DOF - 3 for rotation and 2 for translation, meaning we can only retrieve the direction of the camera pose in frame1 (scale is lost). Thus an assumption is made that the camera pose for frame1 is at unit length wrt camera pose for frame0.   
The relative orientation obtained for the susequent frames is then stacked together to create a complete trajectory.

## Trajectory computed on KITTI dataset

<p align="center">
  <img width="600" height="200" src=" !https://user-images.githubusercontent.com/49958651/93733404-370c1080-fba3-11ea-8b80-02dbde98ae35.gif">
</p>

      ![ezgif com-optimize (3)](https://user-images.githubusercontent.com/49958651/93733404-370c1080-fba3-11ea-8b80-02dbde98ae35.gif)

![ezgif com-optimize (1)](https://user-images.githubusercontent.com/49958651/93733010-a84ac400-fba1-11ea-9ecf-f733f601db0d.gif)
