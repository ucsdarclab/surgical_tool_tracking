import time
import rospy
import cv2
import rosbag
import numpy as np
import copy
import os
import sys

from sensor_msgs.msg import Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber

# ROS Topics
left_camera_topic  = '/stereo/left/image'
right_camera_topic = '/stereo/right/image'
robot_joint_topic  = '/dvrk/PSM1/state_joint_current'
robot_gripper_topic = '/dvrk/PSM1/state_jaw_current'

source_dir = '../journal_dataset/'

# main function
if __name__ == "__main__":
    
    # rosbag input
    bag = rosbag.Bag('../journal_dataset/stationary_camera_2020-06-24-15-49-10.bag')

    old_l_img_msg = None
    old_r_img_msg = None
    old_j_msg = None
    old_g_msg = None
    l_img_msg = None
    r_img_msg = None
    j_msg = None
    g_msg = None

    # get first messages for info
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

        # get one complete message set then break
        break

    # output directory for recordings
    video_out_dir = source_dir
    
    # video recording
    record_video = True
    fps = 30
    left_dims = (int(new_left_img.shape[1]), int(new_left_img.shape[0]))
    right_dims = (int(new_right_img.shape[1]), int(new_right_img.shape[0]))
    if (record_video):
        out_file = video_out_dir + 'left_video.mp4'
        left_video_out  = cv2.VideoWriter(out_file,  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, left_dims)
        out_file = video_out_dir + 'right_video.mp4'
        right_video_out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, right_dims)

        # rosbag input
    bag = rosbag.Bag('../journal_dataset/stationary_camera_2020-06-24-15-49-10.bag')

    old_l_img_msg = None
    old_r_img_msg = None
    old_j_msg = None
    old_g_msg = None
    l_img_msg = None
    r_img_msg = None
    j_msg = None
    g_msg = None

    # get first messages for info
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
    
        # video recording
        if (record_video):
            #print('img_list[0].shape: {}'.format(img_list[0].shape))
            #print('type(img_list[0]): {}'.format(type(img_list[0])))
            left_video_out.write(new_left_img)
            right_video_out.write(new_right_img)
    
    print('end of bag, closing bag')
    bag.close()
    print('Releasing video capture')
    if (record_video):
        left_video_out.release()
        right_video_out.release()