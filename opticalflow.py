import cv2
import numpy as np

class Track():
    
    ## Kanade-Lucas-Tomasi Optical Flow Technique
    def KLT(self,frame0, frame1, frame0_kpts ,show=False): 
        '''
        Parameters
        ----------
        frame0_kpts : pixel location of sift keypoints in frame0
        show : display tracked keypoints on both frames (optional parameter)

        Returns
        -------
        tracked_points0, tracked_points1: Good tracked points from frame0 to frame1
        '''
        frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        p0 = frame0_kpts.reshape(-1,1,2)        
        lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame0_gray, frame1_gray, p0, None, **lk_params)        
        good0 = p0[st==1]
        good1 = p1[st==1]        
        if show:
            temp_frame0 = frame0.copy() 
            temp_frame1 = frame1.copy()
            for x in range(len(good0)):
                cv2.circle(temp_frame0, (int(good0[x,0]), int(good0[x,1])), 1, (0, 0, 255), -1)
                cv2.circle(temp_frame1, (int(good1[x,0]), int(good1[x,1])), 1, (0, 0, 255), -1)
            self.show(temp_frame0)
            self.show(temp_frame1)
        good0, good1 = (np.array(good0, dtype=np.float32), np.array(good1, dtype=np.float32))
        return good0, good1
    
    def show(self,img):
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
