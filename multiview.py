import numpy as np
import cv2

class MultiView():
    def __init__(self):
        pass
        
    def homogeneous(self,points):
        '''
        Parameters
        ----------
        points : column vector of points in euclidian space
        '''
        return np.hstack([points, np.ones((points.shape[0],1))])
    
    def euclidian(self,points):
        '''
        Parameters
        ----------
        points : column vector of points in homogeneous space
        '''
        if points.shape[1] == 3:
            points[:,0] = points[:,0]/points[:,2]
            points[:,1] = points[:,1]/points[:,2]
            return np.array([points[:,0], points[:,1]]).T
        elif points.shape[1] == 4:
            points[:,0] = points[:,0]/points[:,3]
            points[:,1] = points[:,1]/points[:,3]
            points[:,2] = points[:,2]/points[:,3]
            return np.array([points[:,0], points[:,1], points[:,2]]).T
        
    def transform(self,data,transformation): ## Transformation for 3-dim space
        '''
        Parameters
        ----------
        data : column vector of points (3-dim)
        transformation : transformation vector
        '''
        if data.shape[1] == 3:
            data = self.homogeneous(data)
        bottom_row = np.array([0,0,0,1]).reshape(1,4)
        if transformation.shape == (4,4): ## Homogeneous projection matrix
            pass
        elif transformation.shape == (3,3): ## Rotation matrix
            transformation = np.hstack([transformation, np.zeros((3,1))])
            transformation = np.vstack([transformation, bottom_row])
        elif transformation.shape == (3,1): ## translational vector
            transformation = np.hstack([np.eye(3), transformation])
            transformation = np.vstack([transformation, bottom_row])
        elif transformation.shape == (3,4): ## Projection matrix
            transformation = np.vstack([transformation, bottom_row])
        transformed_data = np.dot(transformation, data.T).T
        transformed_data = self.euclidian(transformed_data)
        return transformed_data       
        
    def vector_magnitude(self,vec):
        return np.sqrt(np.sum(vec*vec))    
    
    def get_proj_matrix(self,K,R=np.eye(3),t=np.zeros([3,1])): ## P = KR[I3 - X0]
        KR = np.dot(K,R)
        X0 = np.dot(KR, t)
        return np.hstack([KR, X0])    
    
    def triangulate(self,points1, points2, P1, P2):
        pointclouds = cv2.triangulatePoints(P1, P2, points1.T, points2.T).T
        pointclouds = self.euclidian(pointclouds)
        return pointclouds
    
    def rotate_x(self,vector,theta,degrees=False):       ## Rotates 3-D vector around x-axis
        if degrees:
            theta*=np.pi/180
        R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        return np.dot(R,vector.T).T
        
    def rotate_y(self,vector,theta,degrees=False):       ## Rotates 3-D vector around y-axis
        if degrees:
            theta*=np.pi/180
        R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
        return np.dot(R,vector.T).T
    
    def rotate_z(self,vector,theta,degrees=False):       ## Rotates 3-D vector around z-axis
        if degrees:
            theta*=np.pi/180
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        return np.dot(R,vector.T).T