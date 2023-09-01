import time
import cv2
import kornia as K
import kornia.feature as KF
import rosbag
import os
import sys
import numpy as np

from sensor_msgs.msg import Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber

# set random seed
np.random.seed(0)

script_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.dirname(script_path)
if module_path not in sys.path:
    sys.path.append(module_path)

print(module_path)

from RobotLink_kornia import *
from StereoCamera_kornia import *
from ParticleFilter_kornia import *
from probability_functions_kornia import *
from utils_kornia import *

# File inputs
robot_file    = script_path + '/../../fei_dataset/LND.json'
camera_file   = script_path + '/../../fei_dataset/camera_calibration.yaml'
hand_eye_file = script_path + '/../../fei_dataset/handeye.yaml'

# ROS Topics
left_camera_topic  = '/stereo/left/image'
right_camera_topic = '/stereo/right/image'
robot_joint_topic  = '/dvrk/PSM2/state_joint_current'
robot_gripper_topic = '/dvrk/PSM2/state_jaw_current'

source_dir = 'kornia_dev/fei_ref_data/'

# crop parameters
in_file = source_dir + 'crop_scale.npy'
crop_scale = np.load(in_file)
print('crop_scale: {}'.format(crop_scale))

# Load kornia model
model = KF.SOLD2(pretrained=True, config=None)

robot_arm = RobotLink(robot_file, use_dh_offset=False) # position / orientation in Meters
cam = StereoCamera(camera_file, rectify = True, crop_scale = crop_scale, downscale_factor = 2, scale_baseline=1e-3)

# Load hand-eye transform 
# originally in M
f = open(hand_eye_file)
hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

cam_T_b = np.eye(4)
cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM2_tvec'])/1000.0 # convert to mm
cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM2_rvec'])

# Main loop:
#rate = rospy.Rate(30) # 30hz
prev_joint_angles = None

bag = rosbag.Bag('../fei_dataset/volume_4points_t2.bag')

old_l_img_msg = None
old_r_img_msg = None
old_j_msg = None
old_g_msg = None
l_img_msg = None
r_img_msg = None
j_msg = None
g_msg = None

msg_counter = 0

for topic, msg, t in bag.read_messages(topics=[left_camera_topic, right_camera_topic, robot_joint_topic, robot_gripper_topic]):

    if topic == '/stereo/left/image':
        old_l_img_msg = copy.deepcopy(l_img_msg)
        l_img_msg = copy.deepcopy(msg)
    if topic == '/stereo/right/image':
        old_r_img_msg = copy.deepcopy(r_img_msg)
        r_img_msg = copy.deepcopy(msg)
    if topic == '/dvrk/PSM2/state_joint_current':
        j_msg = copy.deepcopy(msg)
    if topic == '/dvrk/PSM2/state_jaw_current':
        g_msg = copy.deepcopy(msg)
    
    try: 
        if ((l_img_msg != None) and (r_img_msg != None)) and ((l_img_msg != old_l_img_msg) or (r_img_msg != old_r_img_msg)) and (j_msg) and (g_msg):
            _cb_left_img  = np.ndarray(shape=(l_img_msg.height, l_img_msg.width, 3), dtype=np.uint8, buffer=l_img_msg.data)
            _cb_right_img = np.ndarray(shape=(r_img_msg.height, r_img_msg.width, 3), dtype=np.uint8, buffer=r_img_msg.data)
            cb_joint_angles = np.array(j_msg.position + g_msg.position)
        else:
            continue
    except:
        continue

    # copy l/r images so not overwritten by callback
    new_left_img = _cb_left_img.copy()
    new_right_img = _cb_right_img.copy()

    # process callback images
    new_left_img, new_right_img = cam.processImage(new_left_img, new_right_img, crop_scale = crop_scale)
    non_annotated_left_img = new_left_img.copy()
    non_annotated_right_img = new_right_img.copy()

    if(msg_counter == 0):
        outfile = source_dir + 'ref_left_img.jpg'
        print(outfile)
        cv2.imwrite(outfile, new_left_img)
        outfile = source_dir + 'ref_right_img.jpg'
        cv2.imwrite(outfile, new_right_img)

    if(msg_counter == 100):
        outfile = source_dir + 'test_left_img.jpg'
        print(outfile)
        cv2.imwrite(outfile, new_left_img)
        outfile = source_dir + 'test_right_img.jpg'
        cv2.imwrite(outfile, new_right_img)
        break

    msg_counter += 1