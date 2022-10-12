from bvh import Bvh
import config

bvh_file = 'bvh/cmu-mocap/data/001/01_01.bvh'

with open(bvh_file) as f:
    source_motion = Bvh(f.read())

print(source_motion.nframes)

print(source_motion.get_joints_names()[15])

print(source_motion.frame_joint_channels(30, 'Head', ['Xrotation', 'Yrotation', 'Zrotation']))
