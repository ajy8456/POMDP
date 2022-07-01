import numpy as np
import pybullet as p



def matrixBase2World(objectUid):
    '''
    Get transformation matrix to the world frame (hardcoded for readability)
    '''
    obj_pos, obj_orn = p.getBasePositionAndOrientation(objectUid)       # Get (x, y, z) position and (quaternion) orientation
    obj_orn = np.reshape(p.getMatrixFromQuaternion(obj_orn), (3, 3))    # Convert (quaternion) to (rotation matrix)
    translation_matrix = [[1, 0, 0, obj_pos[0]],                        # Convert (obj_pos) to (homogeneous translation)
                          [0, 1, 0, obj_pos[1]],                                   
                          [0, 0, 1, obj_pos[2]],
                          [0, 0, 0, 1]]  
    rotation_matrix = [[obj_orn[0,0], obj_orn[0,1], obj_orn[0,2], 0],   # Convert (obj_orn) to (homogeneous rotation)
                       [obj_orn[1,0], obj_orn[1,1], obj_orn[1,2], 0],    
                       [obj_orn[2,0], obj_orn[2,1], obj_orn[2,2], 0],
                       [0, 0, 0, 1]]        
    T_world = np.matmul(translation_matrix, rotation_matrix)            # We have to rotate first.

    return T_world



def matrixWorld2Camera(view_matrix):
    '''
    Get transformation to the camera frame (hardcoded for readability)
    '''
    view_matrix = np.transpose(np.reshape(view_matrix, (4, 4)))         # Transposing the view matrix for notation reason...
    coord_swap_matrix = np.array([[0, 0, -1, 0],                        # Coordinate swap: view_matrix has -z axis for depth
                                [-1, 0, 0, 0],                          # Swaping to x for depth  (-z -> x), (-x -> y), (y -> z)
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])
    T_camera = np.matmul(coord_swap_matrix, view_matrix)

    return T_camera