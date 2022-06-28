import os
import json
from tqdm import tqdm
import numpy as np
import pybullet as pb
import open3d as o3d
import sys
from contextlib import contextmanager

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

directory_path = 'D:/datasets/partnet-mobility-v0/'
physicsClient = pb.connect(pb.DIRECT)


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def parse_json(parts, part_list):
    for part in parts:
        if 'children' in part:
            part_children = part['children']
            parse_json(part_children, part_list)
        else:
            part_list.append(part)


error_file = open('error_file.txt', "w")

for data_num in tqdm(os.listdir(directory_path)):
    isExist = os.path.exists(directory_path + data_num + '/point_sample/ply-10000.ply')

    if isExist:
        NPYisExist = os.path.exists(directory_path + data_num + '/point_sample/pc_save.npy')
        if NPYisExist:
            continue
        with suppress_stdout():
            try:
                ArticulatedObjId = pb.loadURDF(directory_path + data_num + '/mobility.urdf',
                                               [0, 0, 0], [0, 0, 0, 1], useFixedBase=1)
            except:
                error_file.write(str(data_num) + "\n")
                continue

        URDFInfo = []
        for jointId in range(pb.getNumJoints(ArticulatedObjId)):
            jointInfo = pb.getCollisionShapeData(ArticulatedObjId, jointId)
            for part in jointInfo:
                URDFInfo.append([jointId, part])

        with open(directory_path + data_num + '/result.json') as f:
            json_file = json.load(f)

        PartInfo = []
        parse_json(json_file, PartInfo)

        segment_match = {}
        for j in PartInfo:
            for i in URDFInfo:
                for k in j['objs'][0]:
                    result = i[1][4].decode('utf-8').find(k + '.')
                    if result != -1:
                        segment_match[j['id']] = i[0]

        SegmentLabelOrig = open(directory_path + data_num + '/point_sample/label-10000.txt', "r")
        SegmentLabelCopy = open(directory_path + data_num + '/point_sample/label-10000-urdf.txt', "w")

        while True:
            line = SegmentLabelOrig.readline()
            if not line:
                break
            if int(line) in segment_match.keys():
                SegmentLabelCopy.write(str(segment_match[int(line)]) + "\n")
            else:
                SegmentLabelCopy.write(str(-1) + "\n")

        SegmentLabelOrig.close()
        SegmentLabelCopy.close()

        pc_xyz = np.empty((0, 3), dtype=float)
        PCOrig = open(directory_path + data_num + '/point_sample/pts-10000.txt', "r")
        while True:
            line = PCOrig.readline()
            if not line:
                break
            pc_xyz = np.append(pc_xyz, np.array([line.split()], dtype=float) * 0.1, axis=0)
        PCOrig.close()

        pc_seg = np.empty((0, 1), dtype=float)
        SegmentLabel = open(directory_path + data_num + '/point_sample/label-10000-urdf.txt', "r")
        while True:
            line = SegmentLabel.readline()
            if not line:
                break
            pc_seg = np.append(pc_seg, np.array([line.split()], dtype=float), axis=0)
        SegmentLabel.close()

        pc_seg_dict = {}
        unique_element = np.unique(pc_seg)
        unique_element = unique_element[unique_element != -1]
        for i in unique_element:
            pc_seg_dict[i] = pc_xyz[(pc_seg == i).reshape(-1)]

        pc_save_file = {'xyz': pc_xyz, 'pc_segments': pc_seg_dict}
        np.save(directory_path + data_num + '/point_sample/pc_save', pc_save_file)

    else:
        # os.makedirs(directory_path + data_num + '/point_sample/')
        with suppress_stdout():
            try:
                ArticulatedObjId = pb.loadURDF(directory_path + data_num + '/mobility.urdf',
                                               [0, 0, 0], [0, 0, 0, 1], useFixedBase=1)
            except:
                error_file.write(str(data_num) + "\n")
                continue

        URDFInfo = []
        for jointId in range(pb.getNumJoints(ArticulatedObjId)):
            jointInfo = pb.getCollisionShapeData(ArticulatedObjId, jointId)
            for part in jointInfo:
                URDFInfo.append([jointId, part])

        TotalVolume = 0
        for i in URDFInfo:
            mesh = o3d.io.read_triangle_mesh(i[1][4].decode('UTF-8'))
            aabb = mesh.get_axis_aligned_bounding_box()
            TotalVolume += np.prod(aabb.max_bound - aabb.min_bound)

        pc_xyz = np.empty((0, 3), dtype=float)
        pc_seg = np.empty((0, 1), dtype=float)
        pc_seg_dict = {}
        for i in URDFInfo:
            mesh = o3d.io.read_triangle_mesh(i[1][4].decode('UTF-8'))
            if np.asarray(mesh.triangles).shape[0] >= 8:
                aabb = mesh.get_axis_aligned_bounding_box()
                LocalVolume = np.prod(aabb.max_bound - aabb.min_bound)

                PointsNum = max(int(5000 * (LocalVolume / TotalVolume)), 1000)
                pcd = mesh.sample_points_poisson_disk(number_of_points=PointsNum, init_factor=2)
                pcd = pcd.voxel_down_sample(voxel_size=0.025)

                new_pc = np.asarray(pcd.points)
                if int(i[0]) in pc_seg_dict.keys():
                    pc_seg_dict[int(i[0])] = np.append(pc_seg_dict[int(i[0])], new_pc, axis=0)
                else:
                    pc_seg_dict[int(i[0])] = new_pc
                pc_xyz = np.append(pc_xyz, new_pc, axis=0)
                pc_seg = np.append(pc_seg, np.ones((new_pc.shape[0], 1)) * int(i[0]), axis=0)

        np.savetxt(directory_path + data_num + '/point_sample/pts-10000.txt', pc_xyz, delimiter=' ', fmt='%.6f')
        np.savetxt(directory_path + data_num + '/point_sample/label-10000-modified.txt', pc_seg, delimiter=' ',
                   fmt='%d')
        pc_xyz = pc_xyz * 0.1
        for i in pc_seg_dict:
            pc_seg_dict[i] = pc_seg_dict[i] * 0.1
        pc_save_file = {'xyz': pc_xyz, 'pc_segments': pc_seg_dict}
        np.save(directory_path + data_num + '/point_sample/pc_save', pc_save_file)

error_file.close()
