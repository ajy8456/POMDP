import os
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

        # If mesh has more than 8 triangles
        if np.asarray(mesh.triangles).shape[0] >= 8:

            # 4-a. Calculate local volume (part volume)
            aabb = mesh.get_axis_aligned_bounding_box()
            LocalVolume = np.prod(aabb.max_bound - aabb.min_bound)

            # 4-b. Get point cloud from the part given number of points
            PointsNum = max(int(5000 * (LocalVolume / TotalVolume)), 1000)
            pcd_part = mesh.sample_points_poisson_disk(number_of_points=PointsNum, init_factor=2)
            # pcd = pcd.voxel_down_sample(voxel_size=0.025)

            # 4-c. Convert pointcloud to numpy array and concatenates all parts
            pc_new = np.asarray(pcd_part.points)
            pc_xyz = np.append(pc_xyz, pc_new, axis=0)

    # Convert numpy pointcloud back to o3d type
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)

    return pcd


def removeHiddenPoints(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    '''
    Remove Hidden Points from Open3D pointcloud
    '''

    # TODO: choose exact parameters for removal.
    diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    
    return pcd


def pcBase2World(pcd, g_pos, g_R):
    # Apply transform
    pcd.translate(g_pos, relative=False)
    pcd.rotate(g_R, center=g_pos)
    

def pcWorld2Camera(pcd, c_pos, c_R):
    # Get inverse of camera state
    T = -c_pos
    R = np.linalg.pinv(c_R) 

    # Apply transform
    pcd.translate(T, relative=True)
    pcd.rotate(R)

def instanceSegmentPointCloud(pcd: o3d.geometry.PointCloud, instanceId) -> dict:
    '''
    Add segment id to Open3D pointcloud. Return pointcloud converted to dictionary format.
    '''
    # Convert pointcloud into dictionary
    pc_seg_dict = dict()
    for point in pcd.points:
        # Type: { (x, y, z): instanceId, ... }
        key = tuple(point)
        value = instanceId
        pc_seg_dict[key] = value

    return pc_seg_dict

def visualizePointCloud(pcd):
    
    # o3d.visualization.draw_geometries([pcd], width=480, height=360)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([p[0] for p in pcd.points], [p[1] for p in pcd.points], [p[2] for p in pcd.points], s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show()



if __name__=="__main__":
    p.connect(p.GUI)

    pybullet_data.getDataPath()

    # Load URDF (articulated -> more than 1 joint)
    objId = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbCrackerBox', "model.urdf"), 
                                basePosition=[0, 0, 0], 
                                baseOrientation=[0, 0, 0, 1],
                                useFixedBase=1)    

    # Get instance segmented pointcloud
    pcd = generatePointCloud(objId)
    # pcd = removeHiddenPoints(pcd)

    # Plot pointcloud
    visualizePointCloud(pcd)
            
