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

def checkCollision(obj_list: list, threshold: int=-5e-4) -> bool:
    '''
    Check collision for each pair in the obj_list.

    Params:
    - obj_list: List of uids to compare with each other.
    - threshold: Threshold of collision. Negative when contact.

    Return:
    - boolean
    '''
    for i in range(len(obj_list)):
        for j in range(i, len(obj_list)):
            if i != j:
                contact_points = p.getContactPoints(obj_list[i], obj_list[j])
                contact_dists = [field[8] for field in contact_points]          # field 8: contact distance... negative for penetration
                if len(contact_dists) > 0 and min(contact_dists) <= -1e-3:       # collision when (dist <= -0.001)
                    return True
    else:
        return False
    