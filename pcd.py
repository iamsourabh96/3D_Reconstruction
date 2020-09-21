import numpy as np
import open3d as o3d
import copy
import os

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
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
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
        
    
    def filtr(self,pcd,x_limit=(-3,3),y_limit=(-5,5),z_limit=(0,5),inclusive=True): ## Filters pointcloud given limits in x,y,z direction
        x_min, x_max = x_limit
        y_min, y_max = y_limit
        z_min, z_max = z_limit               
        if not str(type(pcd)) == "<class 'numpy.ndarray'>" :
            pcd = self.pcd2np(pcd)           
        if not inclusive: ## Returns pointcloud outside of the limits provided 
            x0, x1 = pcd[pcd[:,0]<x_min],  pcd[pcd[:,0]>x_max]
            y0, y1 = pcd[pcd[:,1]<y_min], pcd[pcd[:,1]>y_max]   
            z0, z1 = pcd[pcd[:,2]<z_min] , pcd[pcd[:,2]>z_max]
            pcd = np.vstack([x0,x1,y0,y1,z0,z1])
        else: ## Returns pointcloud within the given limits
        
            # p1 = np.array([x_min,y_min,z_min])
            # p2 = np.array([x_min,y_max,z_min])
            # p3 = np.array([x_min,y_max,z_max])
            # p4 = np.array([x_min,y_min,z_max])
            # p5 = np.array([x_max,y_min,z_min])
            # p6 = np.array([x_max,y_max,z_min])
            # p7 = np.array([x_max,y_max,z_max])
            # p8 = np.array([x_max,y_min,z_max])        
            # box = np.vstack([p1,p2,p3,p4,p5,p6,p7,p8]).reshape(-1,3)
            # pcd = np.vstack([pcd, box]) 
            
            pcd = pcd[pcd[:,0]>=x_min]
            pcd = pcd[pcd[:,0]<=x_max]
            pcd = pcd[pcd[:,1]>=y_min]   
            pcd = pcd[pcd[:,1]<=y_max]
            pcd = pcd[pcd[:,2]>=z_min]   
            pcd = pcd[pcd[:,2]<=z_max]            
        return pcd
    
    def create_box(self, x_limits, y_limits, z_limits):
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        z_min, z_max = z_limits 
        p1 = np.array([x_min,y_min,z_min])
        p2 = np.array([x_min,y_max,z_min])
        p3 = np.array([x_min,y_max,z_max])
        p4 = np.array([x_min,y_min,z_max])
        p5 = np.array([x_max,y_min,z_min])
        p6 = np.array([x_max,y_max,z_min])
        p7 = np.array([x_max,y_max,z_max])
        p8 = np.array([x_max,y_min,z_max])        
        box = np.vstack([p1,p2,p3,p4,p5,p6,p7,p8]).reshape(-1,3)
        return box
