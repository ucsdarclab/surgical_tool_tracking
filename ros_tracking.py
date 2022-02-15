import matplotlib.pyplot as plt
import time
import rospy
import cv2
from sensor_msgs.msg import Image, JointState

from message_filters import ApproximateTimeSynchronizer, Subscriber

from RobotLink import *
from StereoCamera import *
from ParticleFilter import *
from probability_functions import *
from utils import *

# File inputs
robot_file    = 'journal_dataset/LND.json'
camera_file   = 'journal_dataset/camera_calibration.yaml'
hand_eye_file = 'journal_dataset/handeye.yaml'

# ROS Topics
left_camera_topic  = '/stereo/left/image'
right_camera_topic = '/stereo/right/image'
robot_joint_topic  = '/dvrk/PSM1/state_joint_current'
robot_gripper_topic = '/dvrk/PSM1/state_jaw_current'

# Globals for callback function
cam = None # cam is global so we can "processImage" in callback
new_cb_data = False
cb_detected_keypoints_l = None
cb_detected_keypoints_r = None
cb_left_img = None
cb_right_img = None
cb_joint_angles = None

# ROS Callback for images and joint observations
def gotData(l_img_msg, r_img_msg, j_msg, g_msg):
    global cam, new_cb_data, cb_detected_keypoints_l, cb_detected_keypoints_r, cb_left_img, cb_right_img, cb_joint_angles
    
    try:
        cb_left_img  = np.ndarray(shape=(l_img_msg.height, l_img_msg.width, 3),
                                      dtype=np.uint8, buffer=l_img_msg.data)
        cb_right_img = np.ndarray(shape=(r_img_msg.height, r_img_msg.width, 3),
                                      dtype=np.uint8, buffer=r_img_msg.data)
        cb_left_img, cb_right_img = cam.processImage(cb_left_img, cb_right_img)
    except:
        return
    cb_detected_keypoints_l, cb_left_img  = segmentColorAndGetKeyPoints(cb_left_img,  draw_contours=True)
    cb_detected_keypoints_r, cb_right_img = segmentColorAndGetKeyPoints(cb_right_img, draw_contours=True)
    cb_joint_angles = np.array(j_msg.position + g_msg.position)
    new_cb_data = True
    

if __name__ == "__main__":
    # Initalize ROS stuff here
    rospy.init_node('robot_tool_tracking', anonymous=True)
    
    l_image_sub = Subscriber(left_camera_topic, Image)
    r_image_sub = Subscriber(right_camera_topic, Image)
    robot_j_sub = Subscriber(robot_joint_topic, JointState)
    gripper_j_sub = Subscriber(robot_gripper_topic, JointState)

    ats = ApproximateTimeSynchronizer([l_image_sub, r_image_sub, robot_j_sub, gripper_j_sub], 
                                      queue_size=5, slop=0.015)
    ats.registerCallback(gotData)


    robot_arm = RobotLink(robot_file)
    cam = StereoCamera(camera_file, rectify=True)

    # Load hand-eye transform 
    f = open(hand_eye_file)
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])


    # Initialize filter
    pf = ParticleFilter(num_states=9, 
                        initialDistributionFunc=sampleNormalDistribution,
                        #motionModelFunc=additiveGaussianNoise, \
                        motionModelFunc=lumpedErrorMotionModel,
                        obsModelFunc=pointFeatureObs,
                        num_particles=200)


    init_kwargs = {
                    "std": np.array([1.0e-3, 1.0e-3, 1.0e-3, # pos
                                    1.0e-2, 1.0e-2, 1.0e-2, # ori
                                    #5.0e-3, 5.0e-3, 0.02
                                    0.0, 0.0, 0.0])   # joints
                  }

    pf.initializeFilter(**init_kwargs)
    
    rospy.loginfo("Initailized particle filter")
       
    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_joint_angles = None

    while not rospy.is_shutdown():
        if new_cb_data:
            start_t = time.time()
            
            # Copy all the new data so they don't get over-written by callback
            new_detected_keypoints_l = np.copy(cb_detected_keypoints_l)
            new_detected_keypoints_r = np.copy(cb_detected_keypoints_r)
            new_left_img = np.copy(cb_left_img)
            new_right_img = np.copy(cb_right_img)
            new_joint_angles = np.copy(cb_joint_angles)
            new_cb_data = False
            
            # First time
            if prev_joint_angles is None:
                prev_joint_angles = new_joint_angles
            
            # Predict Particle Filter
            robot_arm.updateJointAngles(new_joint_angles)
            j_change = new_joint_angles - prev_joint_angles

            std_j = np.abs(j_change)*0.01
            std_j[-3:] = 0.0

            pred_kwargs = {
                            "std_pos": 2.5e-5, 
                            "std_ori": 1.0e-4,
                            "robot_arm": robot_arm, 
                            "std_j": std_j,
                            "nb": 4
                          }
            pf.predictionStep(**pred_kwargs)
            
            
            # Update Particle Filter
            upd_kwargs = {
                            "point_detections": (new_detected_keypoints_l, new_detected_keypoints_r), 
                            "robot_arm": robot_arm, 
                            "cam": cam, 
                            "cam_T_b": cam_T_b,
                            "joint_angle_readings": new_joint_angles,
                            "gamma": 0.15
            }

            pf.updateStep(**upd_kwargs)
            prev_joint_angles = new_joint_angles

            correction_estimation = pf.getMeanParticle()

            rospy.loginfo("Time to predict & update {}".format(time.time() - start_t))

            # Project skeleton
            T = poseToMatrix(correction_estimation[:6])  
            new_joint_angles[-(correction_estimation.shape[0]-6):] += correction_estimation[6:]
            robot_arm.updateJointAngles(new_joint_angles)

            img_list = projectSkeleton(robot_arm.getSkeletonPoints(), np.dot(cam_T_b, T),
                                       [new_left_img, new_right_img], cam.projectPoints)
            
            cv2.imshow("Left Img",  img_list[0])
            cv2.imshow("Right Img", img_list[1])
            cv2.waitKey(1)
            
        
        rate.sleep()

    
