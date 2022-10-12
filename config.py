# import MJCF file
robot_name = "atlas" # amp_humanoid
mocap_name = "cmu"
cmu_class = "001"
motion_name = "01_01"

xml_path = "mjcf/%s.xml"%robot_name
bvh_path = "{}/bvh/cmu-mocap/data/%s/%s.bvh"%(cmu_class,motion_name)
zero_pose_path = "data/zero_pose/%s_zero_pose.npy"%robot_name

source_tpose_path = "data/t_pose/%s_tpose.npy"%mocap_name
target_tpose_path = "data/t_pose/%s_tpose.npy"%robot_name

source_motion_path = "data/source_motion/%s_%s_source.npy"%(mocap_name,motion_name)
target_motion_path = "data/target_motion/%s_%s_%s_target.npy"%(robot_name,mocap_name,motion_name)

retarget_data_path = "retarget_config/retarget_cmu_to_%s.json"%robot_name