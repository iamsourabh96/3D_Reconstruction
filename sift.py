import numpy as np
import cv2


class SIFT():
    def __init__(self):
        self.sift_detector = cv2.xfeatures2d.SIFT_create()
        self.BFMatcher = cv2.BFMatcher()
        
    def features(self,image,coord=False,show=False):
        '''        
        Parameters
        ----------
        coord : returns pixel location of keypoints, optional
            DESCRIPTION. The default is False.
        show : displays sift features with descriptors, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        if not coord: Returns --> kp, des
        else: Returns --> kp, des, coords
        '''
        kp, des = self.sift_detector.detectAndCompute(image, None)
        if show:
            temp_img = cv2.drawKeypoints(image,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.show(temp_img)
        if coord:
            coords= []
            for l in range(len(kp)):
                k = kp[l]
                coords.append(k.pt)
            coords = np.array(coords,dtype=np.float32).reshape(-1,2)
            return kp, des, coords
        return kp, des
        
    def matcher(self,des1,des2,matcher="knn"):
        '''        
        Parameters
        ----------
        des1 : sift feature descriptor from image1
        des2 : sift feature descriptor from image2
        matcher : matcher types --> knn, bf
            DESCRIPTION. The default is "knn".
            knn --> K-nearest neighbours (preferred)
            bf --> Brute Force matcher
            
        Returns
        -------
        matches
        '''
        if matcher == "knn":
            matches = self.BFMatcher.knnMatch(des1, des2, k=2)
            good_matches = []
            ## Ratio Test
            for match1, match2 in matches:
                if match1.distance < 0.75*match2.distance:
                    good_matches.append([match1])
            matches = sorted(good_matches, key = lambda x:x[0].distance)
            return matches        
        elif matcher == "bf":
            bf = cv2.BFMatcher(crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            return matches       
    
    def find_correspondences(self,frame1,frame2,matcher="knn",matches=-1,show=False):
        '''        
        Parameters
        ----------
        image1 : First image
        image2 : Second image
        matcher : matcher types --> knn, bf
            DESCRIPTION. The default is "knn".
            knn --> K-nearest neighbours (preferred)
            bf --> Brute Force matcher            
        matches : number of matches to return (sorted: strongest to weakest)
            DESCRIPTION. The default is -1 --> returns all matches.
            Number should be provided. Eg: 100 --> for first 100 matches
        show : displays correspondances on both images
            DESCRIPTION. The default is False.
            
        Returns
        -------
        pts1: corr points in image1
        pts2: corr points in image2
        '''
        kp1, des1 = self.features(frame1)
        kp2, des2 = self.features(frame2)
        match = self.matcher(des1,des2,matcher)        
        pts1 = []
        pts2 = []
        for idx in match[:matches]:
            pts1.append(kp1[idx[0].queryIdx].pt)
            pts2.append(kp2[idx[0].trainIdx].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2,dtype=np.float32)        
        if show:
            if matcher == "knn":
                sift_matches = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,match[:matches],None,flags=2)
            else:
                sift_matches = cv2.drawMatches(frame1,kp1,frame2,kp2,match[:matches],None,flags=2)
            self.show(sift_matches)
        # return match[:matches], (pts1, pts2)
        return pts1, pts2
    
    def show(self,img):
        cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()