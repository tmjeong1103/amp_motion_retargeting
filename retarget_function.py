import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from math import sqrt

VISUALIZE = False


class FKfunction():

    def get_length(tensor):

        x = -tensor[2]
        y = -tensor[1]
        z = -tensor[0]

        X = x.item()
        Y = y.item()
        Z = z.item()
    
        A= X*X + Y*Y + Z*Z
    
        return sqrt(A)

    def get_length_matrix(float) :
        x = torch.empty(1, 3)

        x[:, 0] = float
        x[:, 1] = float
        x[:, 2] = float

        return x

    def update_global_translation(target_motion, source_motion, target_tpose):

        vec = []
        length = []
        new_pos = [0] * 18

        pos_vec1 = source_motion.local_rotation[..., 13, :] - source_motion.local_rotation[..., 0, :]
        pos_vec2 = source_motion.local_rotation[..., 16, :] - source_motion.local_rotation[..., 13, :]
        pos_vec3 = source_motion.local_rotation[..., 17, :] - source_motion.local_rotation[..., 13, :]
        pos_vec4 = source_motion.local_rotation[..., 19, :] - source_motion.local_rotation[..., 18, :]
        pos_vec5 = source_motion.local_rotation[..., 20, :] - source_motion.local_rotation[..., 19, :]
        pos_vec6 = source_motion.local_rotation[..., 24, :] - source_motion.local_rotation[..., 13, :]
        pos_vec7 = source_motion.local_rotation[..., 26, :] - source_motion.local_rotation[..., 25, :]
        pos_vec8 = source_motion.local_rotation[..., 27, :] - source_motion.local_rotation[..., 26, :]
        pos_vec9 = source_motion.local_rotation[..., 1, :] - source_motion.local_rotation[..., 0, :]
        pos_vec10 = source_motion.local_rotation[..., 3, :] - source_motion.local_rotation[..., 2, :]
        pos_vec11 = source_motion.local_rotation[..., 4, :] - source_motion.local_rotation[..., 3, :]
        pos_vec12 = source_motion.local_rotation[..., 6, :] - source_motion.local_rotation[..., 0, :]
        pos_vec13 = source_motion.local_rotation[..., 8, :] - source_motion.local_rotation[..., 7, :]
        pos_vec14 = source_motion.local_rotation[..., 9, :] - source_motion.local_rotation[..., 8, :]

        #Extra mapping
        pos_vec15 = source_motion.local_rotation[..., 18, :] - source_motion.local_rotation[..., 17, :]
        pos_vec16 = source_motion.local_rotation[..., 25, :] - source_motion.local_rotation[..., 24, :]
        pos_vec17 = source_motion.local_rotation[..., 2, :] - source_motion.local_rotation[..., 1, :]
        pos_vec18 = source_motion.local_rotation[..., 7, :] - source_motion.local_rotation[..., 6, :]

        vec = [pos_vec1, pos_vec2, pos_vec3, pos_vec4, pos_vec5, pos_vec6, pos_vec7, pos_vec8, pos_vec9, pos_vec10, pos_vec11,
            pos_vec12, pos_vec13, pos_vec14, pos_vec15, pos_vec16, pos_vec17, pos_vec18]


        length_vec1 = target_tpose.global_translation[3, :] - target_tpose.global_translation[0, :]
        length_vec2 = target_tpose.global_translation[4, :] - target_tpose.global_translation[3, :]
        length_vec3 = target_tpose.global_translation[11, :] - target_tpose.global_translation[3, :]
        length_vec4 = target_tpose.global_translation[13, :] - target_tpose.global_translation[12, :]
        length_vec5 = target_tpose.global_translation[15, :] - target_tpose.global_translation[13, :]
        length_vec6 = target_tpose.global_translation[5, :] - target_tpose.global_translation[3, :]
        length_vec7 = target_tpose.global_translation[7, :] - target_tpose.global_translation[6, :]
        length_vec8 = target_tpose.global_translation[9, :] - target_tpose.global_translation[7, :]
        length_vec9 = target_tpose.global_translation[24, :] - target_tpose.global_translation[0, :]
        length_vec10 = target_tpose.global_translation[27, :] - target_tpose.global_translation[25, :]
        length_vec11 = target_tpose.global_translation[28, :] - target_tpose.global_translation[27, :]
        length_vec12 = target_tpose.global_translation[18, :] - target_tpose.global_translation[0, :]
        length_vec13 = target_tpose.global_translation[21, :] - target_tpose.global_translation[19, :]
        length_vec14 = target_tpose.global_translation[22, :] - target_tpose.global_translation[21, :]

        #Extra mapping
        length_vec15 = target_tpose.global_translation[12, :] - target_tpose.global_translation[11, :]
        length_vec16 = target_tpose.global_translation[6, :] - target_tpose.global_translation[5, :]
        length_vec17 = target_tpose.global_translation[25, :] - target_tpose.global_translation[24, :]
        length_vec18 = target_tpose.global_translation[19, :] - target_tpose.global_translation[18, :]

        length = [length_vec1, length_vec2, length_vec3, length_vec4, length_vec5, length_vec6, length_vec7, length_vec8, length_vec9,
                length_vec10, length_vec11, length_vec12, length_vec13, length_vec14, length_vec15, length_vec16, length_vec17, length_vec18]


        for i in range(18):
            vec[i] = vec[i][..., :3]
            vec[i] = vec[i] / torch.norm(vec[i], dim=-1, keepdim=True)

            length[i] = get_length(length[i])
            length[i] = get_length_matrix(length[i])

            new_pos[i] = torch.mul(length[i],vec[i])



        #torso, spine
        new_translation_3 = new_pos[0]
        new_translation_4 = new_pos[0] + new_pos[1]
        
        #right_arm
        new_translation_11 = new_pos[0] + new_pos[2]
        new_translation_12 = new_translation_11 + new_pos[14]
        new_translation_13 = new_translation_12 + new_pos[3]
        new_translation_15 = new_translation_13 + new_pos[4]

        #left_arm
        new_translation_5 = new_pos[0] + new_pos[5]
        new_translation_6 = new_translation_5 + new_pos[15]
        new_translation_7 = new_translation_6 + new_pos[6]
        new_translation_9 = new_translation_7 + new_pos[7]

        #right_leg
        new_translation_24 = new_pos[8]
        new_translation_25 = new_translation_24 + new_pos[16]
        new_translation_27 = new_translation_25 + new_pos[9]
        new_translation_28 = new_translation_27 + new_pos[10]

        #left_leg
        new_translation_18 = new_pos[11]
        new_translation_19 = new_translation_18 + new_pos[17]
        new_translation_21 = new_translation_19 + new_pos[12]
        new_translation_22 = new_translation_21 + new_pos[13]


        target_motion.global_translation[...,3,:] = new_translation_3
        target_motion.global_translation[...,4,:] = new_translation_4

        target_motion.global_translation[...,11,:] = new_translation_11
        target_motion.global_translation[...,12,:] = new_translation_12
        target_motion.global_translation[...,13,:] = new_translation_13
        target_motion.global_translation[...,15,:] = new_translation_15

        target_motion.global_translation[...,5,:] = new_translation_5
        target_motion.global_translation[...,6,:] = new_translation_6
        target_motion.global_translation[...,7,:] = new_translation_7
        target_motion.global_translation[...,9,:] = new_translation_9

        target_motion.global_translation[...,24,:] = new_translation_24
        target_motion.global_translation[...,25,:] = new_translation_25
        target_motion.global_translation[...,27,:] = new_translation_27
        target_motion.global_translation[...,28,:] = new_translation_28

        target_motion.global_translation[...,18,:] = new_translation_18
        target_motion.global_translation[...,19,:] = new_translation_19
        target_motion.global_translation[...,21,:] = new_translation_21
        target_motion.global_translation[...,22,:] = new_translation_22

        return target_motion

    """
    v1 = v1[...,:3]
    v2 = v2[...,:3]
    v3 = v3[...,:3]
    v4 = v4[...,:3]
    v5 = v5[...,:3]
    v6 = v6[...,:3]
    v7 = v7[...,:3]
    v8 = v8[...,:3]
    v9 = v9[...,:3]
    v10 = v10[...,:3]
    v11 = v11[...,:3]
    v12 = v12[...,:3]
    v13 = v13[...,:3]
    v14 = v14[...,:3]
    
    #Extra mapping
    v15 = v15[...,:3]
    v16 = v16[...,:3]
    v17 = v17[...,:3]
    v18 = v18[...,:3]

    """


    """
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    v3 = v3 / torch.norm(v3, dim=-1, keepdim=True)
    v4 = v4 / torch.norm(v4, dim=-1, keepdim=True)
    v5 = v5 / torch.norm(v5, dim=-1, keepdim=True)
    v6 = v6 / torch.norm(v6, dim=-1, keepdim=True)
    v7 = v7 / torch.norm(v7, dim=-1, keepdim=True)
    v8 = v8 / torch.norm(v8, dim=-1, keepdim=True)
    v9 = v9 / torch.norm(v9, dim=-1, keepdim=True)
    v10 = v10 / torch.norm(v10, dim=-1, keepdim=True)
    v11 = v11 / torch.norm(v11, dim=-1, keepdim=True)
    v12 = v12 / torch.norm(v12, dim=-1, keepdim=True)
    v13 = v13 / torch.norm(v13, dim=-1, keepdim=True)
    v14 = v14 / torch.norm(v14, dim=-1, keepdim=True)

    #Extra mapping
    v15 = v15 / torch.norm(v1, dim=-1, keepdim=True)
    v16 = v16 / torch.norm(v1, dim=-1, keepdim=True)
    v17 = v17 / torch.norm(v1, dim=-1, keepdim=True)
    v18 = v18 / torch.norm(v1, dim=-1, keepdim=True)
    """

    """
    l1 = get_length(l1)
    l2 = get_length(l2)
    l3 = get_length(l3)
    l4 = get_length(l4)
    l5 = get_length(l5)
    l6 = get_length(l6)
    l7 = get_length(l7)
    l8 = get_length(l8)
    l9 = get_length(l9)
    l10 = get_length(l10)
    l11 = get_length(l11)
    l12 = get_length(l12)
    l13 = get_length(l13)
    l14 = get_length(l14)

    #Extra mapping
    l15 = get_length(l15)
    l16 = get_length(l16)
    l17 = get_length(l17)
    l18 = get_length(l18)
    """

    """
    l1 = get_length_matrix(l1)
    l2 = get_length_matrix(l2)
    l3 = get_length_matrix(l3)
    l4 = get_length_matrix(l4)
    l5 = get_length_matrix(l5)
    l6 = get_length_matrix(l6)
    l7 = get_length_matrix(l7)
    l8 = get_length_matrix(l8)
    l9 = get_length_matrix(l9)
    l10 = get_length_matrix(l10)
    l11 = get_length_matrix(l11)
    l12 = get_length_matrix(l12)
    l13 = get_length_matrix(l13)
    l14 = get_length_matrix(l14)

    #Extra mapping
    l15 = get_length_matrix(l15)
    l16 = get_length_matrix(l16)
    l17 = get_length_matrix(l17)
    l18 = get_length_matrix(l18)
    """

    """
    new_pos1 = torch.mul(length_vec1, pos_vec1)
    new_pos2 = torch.mul(length_vec2, pos_vec2)
    new_pos3 = torch.mul(length_vec3, pos_vec3)
    new_pos4 = torch.mul(length_vec4, pos_vec4)
    new_pos5 = torch.mul(length_vec5, pos_vec5)
    new_pos6 = torch.mul(length_vec6, pos_vec6)
    new_pos7 = torch.mul(length_vec7, pos_vec7)
    new_pos8 = torch.mul(length_vec8, pos_vec8)
    new_pos9 = torch.mul(length_vec9, pos_vec9)
    new_pos10 = torch.mul(length_vec10, pos_vec10)
    new_pos11 = torch.mul(length_vec11, pos_vec11)
    new_pos12 = torch.mul(length_vec12, pos_vec12)
    new_pos13 = torch.mul(length_vec13, pos_vec13)
    new_pos14 = torch.mul(length_vec14, pos_vec14)

    #Extra mapping
    new_pos15 = torch.mul(length_vec15, pos_vec15)
    new_pos16 = torch.mul(length_vec16, pos_vec16)
    new_pos17 = torch.mul(length_vec17, pos_vec17)
    new_pos18 = torch.mul(length_vec18, pos_vec18)

    #torso, spine
    new_translation_3 = new_pos1
    new_translation_4 = new_pos1 + new_pos2

    #right_arm
    new_translation_11 = new_pos1 + new_pos3
    new_translation_12 = new_translation_11 + new_pos15
    new_translation_13 = new_translation_12 + new_pos4
    new_translation_15 = new_translation_13 + new_pos5

    #left_arm
    new_translation_5 = new_pos1 + new_pos6
    new_translation_6 = new_translation_5 + new_pos16
    new_translation_7 = new_translation_6 + new_pos7
    new_translation_9 = new_translation_7 + new_pos8

    #right_leg
    new_translation_24 = new_pos9
    new_translation_25 = new_translation_24 + new_pos17
    new_translation_27 = new_translation_25 + new_pos10
    new_translation_28 = new_translation_27 + new_pos11

    #left_leg
    new_translation_18 = new_pos12
    new_translation_19 = new_translation_18 + new_pos18
    new_translation_21 = new_translation_19 + new_pos13
    new_translation_22 = new_translation_21 + new_pos14

    #6.global_translation update

    target_motion.global_translation[...,3,:] = new_translation_3
    target_motion.global_translation[...,4,:] = new_translation_4

    target_motion.global_translation[...,11,:] = new_translation_11
    target_motion.global_translation[...,12,:] = new_translation_12
    target_motion.global_translation[...,13,:] = new_translation_13
    target_motion.global_translation[...,15,:] = new_translation_15

    target_motion.global_translation[...,5,:] = new_translation_5
    target_motion.global_translation[...,6,:] = new_translation_6
    target_motion.global_translation[...,7,:] = new_translation_7
    target_motion.global_translation[...,9,:] = new_translation_9

    target_motion.global_translation[...,24,:] = new_translation_24
    target_motion.global_translation[...,25,:] = new_translation_25
    target_motion.global_translation[...,27,:] = new_translation_27
    target_motion.global_translation[...,28,:] = new_translation_28

    target_motion.global_translation[...,18,:] = new_translation_18
    target_motion.global_translation[...,19,:] = new_translation_19
    target_motion.global_translation[...,21,:] = new_translation_21
    target_motion.global_translation[...,22,:] = new_translation_22

    return target_motion
    """