# Monocular Visual Odometry
A simple monocular visual odometry implementation with 3 components: 
1. Feature Detection.
2. Obtaining Essential Matrix from point correspondences.
3. Estimating the Relative Orientation from essential matrix.


## Feature Detection and Tracking

My approach uses SIFT feature detectors but this can be easily swapped with other feature detectors like SURF, ORB or FAST corner detector.
The current implementation has two implementations of feature tracking: 
a. Detecting features on every subsequent frames and obtaining correspondences.
b. Using Optical flow to track features to subsequent frames.

The tracking features approach is computationally faster. Tracked features can be lost over the subsequent frames and will have to be reintiallized once they fall below a certain specified threshold. In my approach the features are reinitialized if the number of key points fall below 1500.

If all the point correspondences were perfect, then we would have need only five feature correspondences between two successive frames to estimate motion accurately. However, the feature tracking algorithms are not perfect, and therefore we have several erroneous correspondence. A standard technique of handling outliers when doing model estimation is RANSAC


## Computing Essesntial Matrix

Given the camera intrinsic matrix, Essential Matrix can then be computed from point correspondences using the 5-point algorithm. 
Five points would be sufficient if the point correspondences were perfect but since this is not the case, RANSAC is used to filter outliers.


## Estimating the Relative Orientation from essential matrix

![ezgif com-optimize (3)](https://user-images.githubusercontent.com/49958651/93733404-370c1080-fba3-11ea-8b80-02dbde98ae35.gif)

![ezgif com-optimize (1)](https://user-images.githubusercontent.com/49958651/93733010-a84ac400-fba1-11ea-9ecf-f733f601db0d.gif)
