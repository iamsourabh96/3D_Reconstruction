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
        
    def transform(self,data,transformation):
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
        
        
        


        
    
    
    