# manual annotations for mouse locations during calibration sequences
# used to create training set for YOLOv5
# click on mouse -> saves dvideo mouse picture + dvideo mouse coordinates
 
import numpy as np
import cv2
import csv
import rosbag
import copy
import rospy
import os
import sys

from sensor_msgs.msg import Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber

# function to display the coordinates of
# of the points clicked on the image
def mouse_event(event, x, y, flags, params):
    
    frame_count = int(params[0])

    # checking for mouse clicks
    if (event == cv2.EVENT_MOUSEMOVE) or (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_RBUTTONDOWN):
        
        text_to_save = [frame_count, x, y]
        print(text_to_save)
        with open("keypoint_labels_v2.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(text_to_save)

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
robot_file    = script_path + '/../../journal_dataset/LND.json'
camera_file   = script_path + '/../../journal_dataset/camera_calibration.yaml'
hand_eye_file = script_path + '/../../journal_dataset/handeye.yaml'

# ROS Topics
left_camera_topic  = '/stereo/left/image'
right_camera_topic = '/stereo/right/image'
robot_joint_topic  = '/dvrk/PSM1/state_joint_current'
robot_gripper_topic = '/dvrk/PSM1/state_jaw_current'

# reference image w vs. without contours
draw_contours = False
if (draw_contours):
    source_dir = 'kornia_dev/ref_data/contour/'
else:
    source_dir = 'kornia_dev/ref_data/no_contour/'

# annotate output with detected lines
draw_lines = True

# crop parameters
in_file = source_dir + 'crop_scale.npy'
crop_scale = np.load(in_file)

robot_arm = RobotLink(robot_file, use_dh_offset=False) # position / orientation in Meters
cam = StereoCamera(camera_file, rectify = True, crop_scale = crop_scale, downscale_factor = 2, scale_baseline=1e-3)

# Load hand-eye transform 
# originally in M
f = open(hand_eye_file)
hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

cam_T_b = np.eye(4)
cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0 # convert to mm
cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])

# Main loop:
#rate = rospy.Rate(30) # 30hz
prev_joint_angles = None

old_l_img_msg = None
old_r_img_msg = None
old_j_msg = None
old_g_msg = None
l_img_msg = None
r_img_msg = None
j_msg = None
g_msg = None

bag = rosbag.Bag('../journal_dataset/stationary_camera_2020-06-24-15-49-10.bag')

frame_counter = 1

for topic, msg, t in bag.read_messages(topics=[left_camera_topic, right_camera_topic, robot_joint_topic, robot_gripper_topic]):

    if topic == '/stereo/left/image':
        old_l_img_msg = copy.deepcopy(l_img_msg)
        l_img_msg = copy.deepcopy(msg)
    if topic == '/stereo/right/image':
        old_r_img_msg = copy.deepcopy(r_img_msg)
        r_img_msg = copy.deepcopy(msg)
    if topic == '/dvrk/PSM1/state_joint_current':
        j_msg = copy.deepcopy(msg)
    if topic == '/dvrk/PSM1/state_jaw_current':
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
    
    cv2.imshow('right_img', new_left_img)
    print(frame_counter)
    # set mouse callback
    cv2.setMouseCallback('video_frame', mouse_event, param = [frame_counter])
    frame_counter += 1
    # show one frame at a time
    cv2.waitKey(0)