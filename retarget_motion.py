# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import config
import math # added

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive, plot_skeleton_motion

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""

VISUALIZE = False
def Rx(theta):
  return torch.tensor([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
  return torch.tensor([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])

def Rz(theta):
  return torch.tensor([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

# added
def quat2eul(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        euler_angle = [roll_x, pitch_y, yaw_z]
     
        return euler_angle # in radians

def quat2rotm(quat):
    # Extract the values from Q
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = torch.asarray([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


def rotm2quat(rotm):
        #q0 = qw
    t = torch.trace(rotm)
    q = torch.tensor([0.0, 0.0, 0.0, 0.0])

    if(t > 0):
        t = torch.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (rotm[2,1] - rotm[1,2]) * t
        q[1] = (rotm[0,2] - rotm[2,0]) * t
        q[2] = (rotm[1,0] - rotm[0,1]) * t

    else:
        i = 0
        if (rotm[1,1] > rotm[0,0]):
            i = 1
        if (rotm[2,2] > rotm[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = torch.sqrt(rotm[i,i] - rotm[j,j] - rotm[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (rotm[k,j] - rotm[j,k]) * t
        q[j] = (rotm[j,i] + rotm[i,j]) * t
        q[k] = (rotm[k,i] + rotm[i,k]) * t

    return q

def analytic_ik(motion):
    # 00 root
    pelvis_id = motion.skeleton_tree._node_indices["pelvis"]
    # 01 ~ 03 torso
    ltorso_id = motion.skeleton_tree._node_indices["ltorso"]
    mtorso_id = motion.skeleton_tree._node_indices["mtorso"]
    utorso_id = motion.skeleton_tree._node_indices["utorso"]
    # 04 ~ 09 left_arm
    l_clav_id = motion.skeleton_tree._node_indices["l_clav"]
    l_scap_id = motion.skeleton_tree._node_indices["l_scap"]

    l_uarm_id = motion.skeleton_tree._node_indices["l_uarm"]
    l_larm_id = motion.skeleton_tree._node_indices["l_larm"]

    l_farm_id = motion.skeleton_tree._node_indices["l_farm"]
    l_hand_id = motion.skeleton_tree._node_indices["l_hand"]
    # 10 head
    head_id = motion.skeleton_tree._node_indices["head"]
    # 11 ~ 16 right_arm
    r_clav_id = motion.skeleton_tree._node_indices["r_clav"]
    r_scap_id = motion.skeleton_tree._node_indices["r_scap"] 

    r_uarm_id = motion.skeleton_tree._node_indices["r_uarm"]
    r_larm_id = motion.skeleton_tree._node_indices["r_larm"]

    r_farm_id = motion.skeleton_tree._node_indices["r_farm"]
    r_hand_id = motion.skeleton_tree._node_indices["r_hand"]
    # 17 ~22 left_leg
    l_uglut_id = motion.skeleton_tree._node_indices["l_uglut"]
    l_lglut_id = motion.skeleton_tree._node_indices["l_lglut"]

    l_uleg_id = motion.skeleton_tree._node_indices["l_uleg"]
    l_lleg_id = motion.skeleton_tree._node_indices["l_lleg"]

    l_talus_id = motion.skeleton_tree._node_indices["l_talus"]
    l_foot_id = motion.skeleton_tree._node_indices["l_foot"]
    # 23 ~ 28 right_leg
    r_uglut_id = motion.skeleton_tree._node_indices["r_uglut"]
    r_lglut_id = motion.skeleton_tree._node_indices["r_lglut"]
    
    r_uleg_id = motion.skeleton_tree._node_indices["r_uleg"]
    r_lleg_id = motion.skeleton_tree._node_indices["r_lleg"]

    r_talus_id = motion.skeleton_tree._node_indices["r_talus"]
    r_foot_id = motion.skeleton_tree._node_indices["r_foot"]

    device = motion.global_translation.device

    # pelvis, root
    pelvis_quat = motion.local_rotation[..., pelvis_id, :]
    # quaternion to euler angle
    motion_size = pelvis_quat[:,0].size()

    # step1) torso part
    ltorso_quat = motion.local_rotation[..., ltorso_id, :] # angle axis [0 0 1]
    mtorso_quat = motion.local_rotation[..., mtorso_id, :] # angle axis [0 1 0]
    utorso_quat = motion.local_rotation[..., utorso_id, :] # angle axis [1 0 0]
    for tick in range(motion_size[0]-1):
        ltorso_curr_quat = ltorso_quat[tick,:]
        euler_angle = quat2eul(ltorso_curr_quat)

        # rotate theta for each axis
        roll_theta = euler_angle[0]
        pitch_theta = euler_angle[1]
        yaw_theta = euler_angle[2]

        # convert quaternion to rotation matrix
        child_name = 'ltorso'
        curr_idx = motion.skeleton_tree._node_indices[child_name]
        parent_idx = motion.skeleton_tree._parent_indices[curr_idx]
        parent_quat = motion.local_rotation[tick, parent_idx, :]
        parent_rotm = quat2rotm(parent_quat)        
        # rotation components of torso
        rotx = Rx(roll_theta)
        inv_rotx = torch.transpose(rotx,0,1)
        roty = Ry(pitch_theta)
        inv_roty = torch.transpose(roty,0,1)
        rotz = Rz(yaw_theta)
        inv_rotz = torch.transpose(rotz,0,1)

        # utorso [1 0 0]
        utorso_quat[tick,:] = ltorso_quat[tick,:]
        # mtorso [0 1 0]
        mtorso_rotm = inv_rotx*quat2rotm(utorso_quat[tick,:])
        mtorso_quat[tick,:] = rotm2quat(mtorso_rotm)
        # ltorso [0 0 1]
        ltorso_rotm = inv_roty*quat2rotm(mtorso_quat[tick,:])
        ltorso_quat[tick,:] = rotm2quat(ltorso_rotm)

        # # ltorso [0 0 1]
        # ltorso_rotm = rotz*parent_rotm
        # ltorso_quat[tick,:] = rotm2quat(ltorso_rotm)
        # # mtorso [0 1 0]
        
        # mtorso_quat[tick,:] = rotm2quat(mtorso_rotm)
        # # utorso [1 0 0]
        # utorso_rotm = rotx*mtorso_rotm
        # utorso_quat[tick,:] = rotm2quat(utorso_rotm)

        # update
        motion.local_rotation[..., ltorso_id, :] = ltorso_quat[tick,:]
        motion.local_rotation[..., mtorso_id, :] = mtorso_quat[tick,:]        
        motion.local_rotation[..., utorso_id, :] = utorso_quat[tick,:]

    # 10 head
    head_id = motion.skeleton_tree._node_indices["head"]

    # step2) left_arm - shoulder part  
    l_clav_quat = motion.local_rotation[..., l_clav_id, :] # angle axis [0 0.500000 0.866025]
    l_scap_quat = motion.local_rotation[..., l_scap_id, :] # angle axis [1 0 0]    
    for tick in range(motion_size[0]-1):
        l_clav_curr_quat = l_clav_quat[tick,:]
        euler_angle = quat2eul(l_clav_curr_quat)

        # rotate theta for each axis
        roll_theta = euler_angle[0]
        pitch_theta = euler_angle[1]
        yaw_theta = euler_angle[2]

        # convert quaternion to rotation matrix
        child_name = 'l_clav'
        curr_idx = motion.skeleton_tree._node_indices[child_name]
        parent_idx = motion.skeleton_tree._parent_indices[curr_idx]
        parent_quat = motion.local_rotation[tick, parent_idx, :]
        parent_rotm = quat2rotm(parent_quat)        
        
        # rotation components of shoulder
        rotx = Rx(roll_theta)
        inv_rotx = torch.transpose(rotx,0,1)
        roty = Ry(pitch_theta)
        rotz = Rz(yaw_theta)

        # l_scap [1 0 0]
        l_scap_quat[tick,:] = l_clav_curr_quat

        # l_clav [0 0.500000 0.866025]
        l_clav_rotm = inv_rotx*quat2rotm(l_scap_quat[tick,:])
        l_clav_quat[tick,:] = rotm2quat(l_clav_rotm)     

        # update
        motion.local_rotation[..., l_scap_id, :] = l_scap_quat[tick,:]
        motion.local_rotation[..., l_clav_id, :] = l_clav_quat[tick,:]

    # step3) left_arm - arm part
    l_uarm_quat = motion.local_rotation[..., l_uarm_id, :] # angle axis [0 1 0]
    l_larm_quat = motion.local_rotation[..., l_larm_id, :] # angle axis [1 0 0] 
    for tick in range(motion_size[0]-1):
        l_uarm_curr_quat = l_uarm_quat[tick,:]
        euler_angle = quat2eul(l_uarm_curr_quat)

        # rotate theta for each axis
        roll_theta = euler_angle[0]
        pitch_theta = euler_angle[1]
        yaw_theta = euler_angle[2]

        # convert quaternion to rotation matrix
        child_name = 'l_uarm'
        curr_idx = motion.skeleton_tree._node_indices[child_name]
        parent_idx = motion.skeleton_tree._parent_indices[curr_idx]
        parent_quat = motion.local_rotation[tick, parent_idx, :]
        parent_rotm = quat2rotm(parent_quat)        
        
        # rotation components of shoulder
        rotx = Rx(roll_theta)
        inv_rotx = torch.transpose(rotx,0,1)
        roty = Ry(pitch_theta)
        inv_roty = torch.transpose(roty,0,1)
        rotz = Rz(yaw_theta)
        inv_rotz = torch.transpose(rotz,0,1)

        # l_larm [0 1 0]
        l_larm_quat[tick,:] = l_uarm_curr_quat

        # l_uarm [1 0 0]
        l_uarm_rotm = inv_roty*quat2rotm(l_larm_quat[tick,:])
        l_uarm_quat[tick,:] = rotm2quat(l_uarm_rotm)     

        # update
        motion.local_rotation[..., l_uarm_id, :] = l_uarm_quat[tick,:]
        motion.local_rotation[..., l_larm_id, :] = l_larm_quat[tick,:]


def main():
    # load retarget config
    with open(config.retarget_data_path) as f:
        retarget_data = json.load(f)

    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(config.source_tpose_path)
    if VISUALIZE:
        plot_skeleton_state(source_tpose)

    target_tpose = SkeletonState.from_file(config.target_tpose_path)
    if VISUALIZE:
        plot_skeleton_state(target_tpose)

    # load and visualize source motion sequence
    source_motion = SkeletonMotion.from_file(config.source_motion_path)
    if VISUALIZE:
        plot_skeleton_motion_interactive(source_motion)

    # parse data from retarget config
    joi_mapping = retarget_data["joi_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = source_motion.local_rotation.shape[0]
        
    local_rotation = source_motion.local_rotation
    root_translation = source_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    new_sk_state = SkeletonState.from_rotation_and_root_translation(source_motion.skeleton_tree,\
         local_rotation, root_translation, is_local=True)
    source_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=source_motion.fps)
    # plot_skeleton_motion_interactive(source_motion)

    # run retargeting
    # modify: scale is set automatically
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=joi_mapping,
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )
    # plot_skeleton_motion_interactive(test_target_motion)

    # analytic_ik(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h
    
    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_data["root_height_offset"]
    root_translation[:, 2] += root_height_offset
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(config.target_motion_path)

    # visualize retargeted motion
    plot_skeleton_motion_interactive(target_motion)

    return

if __name__ == '__main__':
    main()