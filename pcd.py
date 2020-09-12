import numpy as np
import open3d as o3d
import copy

class PCD():   
    
    def np2pcd(self,numpyData): ## Converts numpy to pcd format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpyData)
        return pcd    

    def pcd2np(self,pcd): ## Converts pcd to numpy format
        return np.asarray(pcd.points)
 
    def viz(self,data,color=False,axis=False):
        '''
        Parameters
        ----------
        data : numpy or pcd data format. Should be a list.
            DESCRIPTION: Visualization for 3D pointclouds
        color : paints pointcloud uniformly by the color provided (optional argument))
        axis : creates coordinate axis at origin (optional argument)
        
        Returns
        -------
        None.
        '''
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        for x in range(len(data)):    
            if str(type(data[x])) == "<class 'numpy.ndarray'>" :
                 data[x] = self.np2pcd(data[x])                 
        if axis:
            data+=[mesh_frame]          
        if color:
            for x in range(len(color)):
                data[x].paint_uniform_color(color[x])             
        o3d.visualization.draw_geometries(data)
    
    def create_axis(self, size=0.02, loc=[0,0,0]): ## Creates coordinate axis at location provided
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=loc)
        return mesh_frame
    
    def transform(self,data,transformation):
        '''
        Parameters
        ----------
        data : data in Open3d format
        transformation : transformation vector
        '''
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
        transformed_data = data.transform(transformation)
        return transformed_data  
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
