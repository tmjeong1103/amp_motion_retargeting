from colorsys import rgb_to_yiq
import sys
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as Rot

deg2rad = np.pi / 180.0
R1 = Rot.from_euler('zyx', [0.0,0.0,0.0], degrees=True).as_matrix()
R2 = Rot.from_euler('xyz', [0.0,0.0,0.0], degrees=True).as_matrix()
def bvh_to_array(bvh_file, root_joint, data_type=np.float32, debug=True, channels = ["Xrotation","Yrotation","Zrotation"],root_xyz_channels = ["Xposition","Yposition","Zposition"],offset_order=[0,1,2]):
    # Open bvh file
    with open(bvh_file) as file:
        print("Opening {}".format(bvh_file))
        print("Might take few seconds. (TODO: improve third party backend)")
        mocapfile = file.read()
        mocap = Bvh(mocapfile) # parse

        # Extract joint name, and initialize reference and parent array
        joint_names = mocap.get_joints_names()
        joint_ref = [None]+joint_names[:]
        joint_parent = len(joint_names)*[-1]

        # Buffers to store
        frame_joint_trans = []
        joint_offsets = []
        root_xyz = []

        for j_idx, joint_name in enumerate(joint_names):
            joint = mocap.get_joint(joint_name)
            
            print(joint_name)

            # Root check
            if j_idx == 0 and not joint_name == root_joint:
                raise ValueError("The name of the root node in the tree does not match the declared root name")
            elif j_idx == 0 and joint_name == root_joint:
                root_xyz = np.array(mocap.frames_joint_channels(joint_name, root_xyz_channels),dtype=data_type)
                root_xyz = np.apply_along_axis(lambda x: R1.dot(x),1,root_xyz)
                root_xyz = np.apply_along_axis(lambda x: _frame_to_se(np.concatenate([np.multiply(x,[1.0,1.0,1.0]),np.array([0.0,0.0,0.0],dtype=data_type)])), 1, root_xyz)

            j_o = np.array(joint["OFFSET"],dtype=data_type)
            j_off = np.array(R2.dot(j_o))
            # j_off[1], j_off[2] = j_off[2], j_off[1]
            j_offset = np.array(mocap.nframes * [j_off], dtype=data_type) # n_frames x 3

            if joint_name == "Hips":
                print("Hit")
                rotations = np.array(mocap.frames_joint_channels(joint_name, channels), dtype=data_type) + np.array(mocap.nframes*[[0.0,0.0,0.0]],dtype=np.float32)
                print(rotations)
            else:
                rotations = np.array(mocap.frames_joint_channels(joint_name, channels), dtype=data_type) + np.array(mocap.nframes*[[0.0,0.0,0.0]],dtype=np.float32)
                #rotations = np.array(mocap.frames_joint_channels(joint_name, channels), dtype=data_type) # n_frames x 3

            joint_parent[j_idx] = mocap.joint_parent_index(joint_name)
            rotations = np.array(mocap.frames_joint_channels(joint_name, channels))
            joint_offsets.append(joint["OFFSET"])
            frame_joint_trans.append(np.concatenate([j_offset,rotations],axis=1))
        
        frame_joint_trans = np.array(frame_joint_trans, dtype=data_type)
        joint_offsets = np.array(joint_offsets, dtype=data_type)
        frame_joint_trans = frame_joint_trans.transpose(1,0,2)
        frame_joint_trans = np.apply_along_axis(_frame_to_se, 2, frame_joint_trans)

        if debug:
            print(np.size(frame_joint_trans))
            print("MOCAP INFO")
            print("Filename {}".format(bvh_file))
            print("Total frames: {}".format(mocap.nframes))
            print("FPS: {}".format(int(1/mocap.frame_time)))
            print("Channels: {}".format(channels))
            print("Total joints: {}".format(len(joint_names)))
            print("Joint names: {}".format(joint_names))
            print("Parent names {}".format(list(map(lambda x:joint_ref[x+1],joint_parent))))
            print("Parent indices: {}".format(joint_parent))
            print("Orientation shape: {}".format(np.shape(frame_joint_trans)))
            print("Orientation size: {}".format(np.size(frame_joint_trans)))
            print("Orientation size sanity check {}".format(len(joint_names)*mocap.nframes*2*len(channels)))
            print("Root XYZ size: {}".format(root_xyz.shape))
            print("Offset dim {}".format(joint_offsets.shape))

        return joint_names, joint_parent, frame_joint_trans, root_xyz, int(1/mocap.frame_time)

def _euler_to_rot(x_rot, y_rot, z_rot):
    R = np.zeros((3,3))
    alpha = x_rot * deg2rad
    beta = y_rot * deg2rad
    gamma = z_rot * deg2rad

    # fix = Rot.from_euler('xyz', [0,90,0], degrees=True).as_matrix()
    # R1 = Rot.from_euler('zyx', [z_rot,y_rot,x_rot], degrees=True).as_matrix()

    R[0,0] = np.cos(beta)*np.cos(gamma)
    R[0,1] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
    R[0,2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
    R[1,0] = np.cos(beta)*np.sin(gamma)
    R[1,1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
    R[1,2] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
    R[2,0] = -np.sin(beta)
    R[2,1] = np.sin(alpha)*np.cos(beta)
    R[2,2] = np.cos(alpha)*np.cos(beta)

    return np.array(R)

def _frame_to_se(vec6d):
    x_pos,y_pos,z_pos,x_rot,y_rot,z_rot = vec6d[0],vec6d[1],vec6d[2],vec6d[3],vec6d[4],vec6d[5]
    se = np.zeros((4,4))
    se[3,3] = 1.0
    # se[:3,3] = np.transpose([x_pos,y_pos,z_pos])
    se[:3,3] = np.transpose([x_pos,y_pos,z_pos])
    se[:3,:3] = _euler_to_rot(x_rot, y_rot, z_rot)
    return se