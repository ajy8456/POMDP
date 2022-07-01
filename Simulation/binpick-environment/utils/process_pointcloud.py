import os
from turtle import pos
import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
from pybullet_object_models import ycb_objects
import open3d as o3d


def generatePointCloud(uId) -> o3d.geometry.PointCloud:
    '''
    Generate Open3D pointcloud from pyBullet URDF ID.
    The poincloud has its origin at the base of 
    '''

    MAX_NUM_POINTS = 2500
    MIN_NUM_POINTS = 1000

    # 1. Collect all shape data of parts along links in URDF
    #     Link index description
    #         -1: base
    #          0: link after first joint
    #     
    #     partsInfo = [ part1_info , part2_info , ... ]
    #     Check pybulletQuickStartGuide for the return type of p.getCollisionShapeData()
    partsInfo = []
    for linkId in range(-1, p.getNumJoints(uId)):
        linkInfo = p.getCollisionShapeData(uId, linkId)
        for part in linkInfo:
            partsInfo.append(part)

    # 2. Calcuated total volume of the URDF. The volume is later used to decide the number of points cloud.
    TotalVolume = 0
    for i in partsInfo:
        mesh = o3d.io.read_triangle_mesh(i[4].decode('UTF-8')) # i[4] -> mesh file path.
        aabb = mesh.get_axis_aligned_bounding_box()
        TotalVolume += np.prod(aabb.max_bound - aabb.min_bound)
    
    # 4. Generating pointcloud
    pc_xyz = np.empty((0, 3), dtype=float)
    
    for i in partsInfo:
        # Read mesh file of URDF.
        mesh = o3d.io.read_triangle_mesh(i[4].decode('UTF-8'))

        # If mesh has triangles
        if np.asarray(mesh.triangles).shape[0] >= 1:

            # 4-a. Calculate local volume (part volume)
            aabb = mesh.get_axis_aligned_bounding_box()
            LocalVolume = np.prod(aabb.max_bound - aabb.min_bound)

            # 4-b. Get point cloud from the part given number of points
            PointsNum = max(int(MAX_NUM_POINTS * (LocalVolume / TotalVolume)), MIN_NUM_POINTS)
            pcd_part = mesh.sample_points_poisson_disk(number_of_points=PointsNum, init_factor=2)
            # pcd = pcd.voxel_down_sample(voxel_size=0.025)

            # 4-c. Convert pointcloud to numpy array and concatenates all parts
            pc_new = np.asarray(pcd_part.points)
            pc_xyz = np.concatenate((pc_xyz, pc_new), axis=0)

    # Convert numpy pointcloud back to o3d type
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)

    return pcd
    

def transformPointCloud(pcd, T):
    # Apply transform
    pcd.transform(T)


def mergePointCloud(pcd_list):
    '''
    Merge multiple point clouds in the list
    Reference: https://sungchenhsi.medium.com/adding-pointcloud-to-pointcloud-9bf035707273
    '''
    merged_pcd_numpy = np.empty((0, 3), dtype=float)
    for _pcd in pcd_list:
        _pcd_numpy = np.asarray(_pcd.points)                 # Use np.asarray to avoid meaningless copy
        merged_pcd_numpy = np.concatenate((merged_pcd_numpy, _pcd_numpy), axis=0)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_pcd_numpy)

    return merged_pcd



def removeHiddenPoints(pcd: o3d.geometry.PointCloud, camera: np.array):
    '''
    Remove Hidden Points from Open3D pointcloud
    Reference: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html

    Params:
    - pcd: Point cloud to process
    - camera: The location of the camera. Orientation is not considered.
    '''
    diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    
    return pcd, pt_map


def pointCloud2Dictionary(pcd, pt_map):
    pc_dict = {
        (_p[0], _p[1], _p[2]) : (1 if i in pt_map else 0)
        for (i, _p) in enumerate(np.asarray(pcd.points))    # Use np.asarray to avoid meaningless copy
    }

    return pc_dict


def visualizePointCloud(pcd):
    '''
    Visualize open3d type or dictionary type point cloud
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])

    # Plot dictionary type point cloud
    if isinstance(pcd, dict):
        pos = [ _p for _p, _v in pcd.items() if _v == 1]
        neg = [ _p for _p, _v in pcd.items() if _v == 0]
        ax.scatter([_p[0] for _p in pos], [_p[1] for _p in pos], [_p[2] for _p in pos], s=0.2)
        ax.scatter([_p[0] for _p in neg], [_p[1] for _p in neg], [_p[2] for _p in neg], s=0.2, c="r")
    # Plot open3d type point cloud
    elif isinstance(pcd, o3d.geometry.PointCloud):
        # o3d.visualization.draw_geometries([pcd], width=480, height=360)
        ax.scatter([p[0] for p in pcd.points], [p[1] for p in pcd.points], [p[2] for p in pcd.points], s=0.2)  
    else:
        print("Unidenified type to visualize")

    plt.show()


if __name__=="__main__":
    p.connect(p.GUI)

    pybullet_data.getDataPath()
    print(ycb_objects.getDataPath())

    # Load URDF (articulated -> more than 1 joint)
    objId = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"), 
                                basePosition=[0, 0, 0], 
                                baseOrientation=[0, 0, 0, 1],
                                useFixedBase=1)    

    # Get instance segmented pointcloud
    pcd = generatePointCloud(objId)
    # pcd = removeHiddenPoints(pcd)

    # Plot pointcloud
    visualizePointCloud(pcd)
            
