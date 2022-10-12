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

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

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

    # run retargeting
    # modify: scale is set automatically
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=joi_mapping,
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)
    plot_skeleton_motion_interactive(target_motion)

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