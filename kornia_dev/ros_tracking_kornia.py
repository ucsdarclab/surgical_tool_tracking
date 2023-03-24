import time
import rospy
import cv2
import kornia as K
import kornia.feature as KF

from sensor_msgs.msg import Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber

import os
import sys

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

# Globals for callback function
global _cb_left_img
_cb_left_img = None
global _cb_right_img 
_cb_right_img = None
global cb_joint_angles 
cb_joint_angles = None
global new_cb_data 
new_cb_data = False

# ROS Callback for images and joint observations
def gotData(l_img_msg, r_img_msg, j_msg, g_msg):
    
    global _cb_left_img
    global _cb_right_img
    global cb_joint_angles
    global new_cb_data
    
    try:
        _cb_left_img  = np.ndarray(shape=(l_img_msg.height, l_img_msg.width, 3),
                                      dtype=np.uint8, buffer=l_img_msg.data)
        _cb_right_img = np.ndarray(shape=(r_img_msg.height, r_img_msg.width, 3),
                                      dtype=np.uint8, buffer=r_img_msg.data)
    except:
        return
    
    cb_joint_angles = np.array(j_msg.position + g_msg.position)
    new_cb_data = True


# main function
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

    # crop parameters
    orig_ref_dims = np.load('orig_ref_dims.npy')
    crop_ref_dims = np.load('crop_ref_dims.npy') # array([ 405, 720])

    # reference lines
    crop_ref_lines_l = torch.as_tensor(np.load('crop_ref_lines_l.npy')) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]
    crop_ref_lines_r = torch.as_tensor(np.load('crop_ref_lines_l.npy')) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]

    # reference images
    # left camera
    crop_ref_l = np.load('crop_ref_l.npy') # (405, 720, 3) RGB uint8
    crop_ref_l = K.image_to_tensor(crop_ref_l).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
    crop_ref_l = K.color.rgb_to_grayscale(crop_ref_l) # [0, 1] torch.Size([1, 720, 1080]) torch.float32
    # right camera
    crop_ref_r = np.load('crop_ref_r.npy') # (1080, 1920, 3) RGB uint8
    crop_ref_r = K.image_to_tensor(crop_ref_r).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
    crop_ref_r = K.color.rgb_to_grayscale(crop_ref_r) # [0, 1] torch.Size([1, 720, 1080]) torch.float32

    # Load kornia model
    model = KF.SOLD2(pretrained=True, config=None)

    # parameters for shaft detection
    canny_params = {
        'use_canny': False,
        'hough_rho_accumulator': 5.0,
        'hough_theta_accumulator': 0.09,
        'hough_vote_threshold': 100,
        'rho_cluster_distance': 5.0,
        'theta_cluster_distance': 0.09
    }

    kornia_params = {
        'use_kornia': True,
        'endpoints_to_polar': False,
        'use_endpoint_intensities_only': False,
        'endpoint_intensities_to_polar': False,
        'search_radius': 25.0,
        'intensity_params': {
            'use_metric': 'mean',
            'mean': 0,
            'std': 1.0,
            'pct': 10.0
        },
        'ransac_params': {
            'min_samples': 3.0,
            'residual_threshold': 0.75,
            'max_trials': 100,
            'crop_ref_dims': crop_ref_dims
        },
        'use_line_intensities_only': False,
        'line_intensities_to_polar': True
    } 

    robot_arm = RobotLink(robot_file, use_dh_offset=False) # position / orientation in Meters
    cam = StereoCamera(camera_file, rectify = True, orig_ref_dims = orig_ref_dims, crop_ref_dims = crop_ref_dims)

    # Load hand-eye transform 
    # originally in M
    f = open(hand_eye_file)
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0 # convert to mm
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])


    # Initialize filter
    pf = ParticleFilter(num_states=9, 
                        initialDistributionFunc=sampleNormalDistribution,
                        #motionModelFunc=additiveGaussianNoise, \
                        motionModelFunc=lumpedErrorMotionModel,
                        #obsModelFunc=pointFeatureObs,
                        obsModelFunc=[
                                    pointFeatureObs, 
                                    shaftFeatureObs_kornia
                                    ],
                        num_particles=200)


    init_kwargs = {
                    "std": np.array([1.0e-3, 1.0e-3, 1.0e-3, # pos # in M i.e. 1x10^-3 M
                                    1.0e-2, 1.0e-2, 1.0e-2, # ori
                                    #5.0e-3, 5.0e-3, 0.02
                                    0.0, 0.0, 0.0])   # joints
                  }

    pf.initializeFilter(**init_kwargs)
    
    rospy.loginfo("Initialized particle filter")
       
    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_joint_angles = None

    while not rospy.is_shutdown():
        if new_cb_data:
            start_t = time.time()

            # copy l/r images so not overwritten by callback
            new_left_img = _cb_left_img.copy()
            new_right_img = _cb_right_img.copy()

            # process callback images
            new_left_img, new_right_img = cam.processImage(new_left_img, new_right_img)
            detected_keypoints_l, new_left_img  = segmentColorAndGetKeyPoints(new_left_img,  draw_contours=True)
            new_detected_keypoints_l = np.copy(detected_keypoints_l)
            detected_keypoints_r, new_right_img = segmentColorAndGetKeyPoints(new_right_img, draw_contours=True)
            new_detected_keypoints_r = np.copy(detected_keypoints_r)
            
            #cb_detected_shaftlines_l, _cb_left_img, _cb_left_ref_img  = detectShaftLines(_cb_left_img, crop_ref_l, crop_ref_lines_l, crop_ref_dims, model, cb_use_intensity, cb_intensity_radius) # (returns torch[2, 2, 2] of endpoints, Nx2 array of [rho, theta]), image with line segments drawn, ref image with reference line segments drawn
            #cb_detected_shaftlines_r, _cb_right_img, _cb_right_ref_img = detectShaftLines(_cb_right_img, crop_ref_r, crop_ref_lines_r, crop_ref_dims, model, cb_use_intensity, cb_intensity_radius) # (returns torch[2, 2, 2] of endpoints, Nx2 array of [rho, theta]),image with line segments drawn, ref image with reference line segments drawn
            output_l  = detectShaftLines(new_img = new_left_img, 
                                        ref_img = crop_ref_l,
                                        crop_ref_lines = crop_ref_lines_l,
                                        crop_ref_dims = crop_ref_dims,
                                        model = model,
                                        canny_params = canny_params,
                                        kornia_params = kornia_params)
            output_r  = detectShaftLines(new_img = new_right_img, 
                                        ref_img = crop_ref_r,
                                        crop_ref_lines = crop_ref_lines_r,
                                        crop_ref_dims = crop_ref_dims,
                                        model = model,
                                        canny_params = canny_params,
                                        kornia_params = kornia_params)

            # copy new images to avoid overwriting by callback
            new_left_img  = np.copy(output_l['new_img']) # cropped img w/detected lines
            new_left_ref_img = np.copy(output_l['ref_img']) # cropped img w/ref line segments
            new_right_img = np.copy(output_r['new_img']) # cropped img w/detected lines
            new_right_ref_img = np.copy(output_r['ref_img']) # cropped img w/ref line segments

            # Nx2 array [[rho, theta], [rho, theta], ...]
            new_canny_lines_l = np.copy(output_l['canny_lines']) 
            new_detected_endpoint_lines_l = np.copy(output_l['polar_lines_detected_endpoints']) # Nx2 array [[rho, theta], [rho, theta], ...]
            new_endpoint_clouds_l =  np.copy(output_l['intensity_endpoint_clouds'])
            new_endpoint_cloud_lines_l = np.copy(output_l['intensity_endpoint_lines'])
            new_line_clouds_l = np.copy(output_l['intensity_line_clouds'])
            new_line_cloud_lines_l = np.copy(output_l['intensity_line_lines'])

            new_canny_lines_r = np.copy(output_r['canny_lines']) 
            new_detected_endpoint_lines_r = np.copy(output_r['polar_lines_detected_endpoints']) # Nx2 array [[rho, theta], [rho, theta], ...]
            new_endpoint_clouds_r =  np.copy(output_r['intensity_endpoint_clouds'])
            new_endpoint_cloud_lines_r = np.copy(output_r['intensity_endpoint_lines'])
            new_line_clouds_r = np.copy(output_r['intensity_line_clouds'])
            new_line_cloud_lines_r = np.copy(output_r['intensity_line_lines'])
            
            # copy new joint angles to ensure no overwrite
            new_joint_angles = np.copy(cb_joint_angles)
            
            # update callback flag
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
                            "std_pos": 2.5e-5, # in Meters
                            "std_ori": 1.0e-4,
                            "robot_arm": robot_arm, 
                            "std_j": std_j,
                            "nb": 4
                          }
            pf.predictionStep(**pred_kwargs)
            
            # Update Particle Filter
            upd_args = [  
                        # pointFeatureObs arguments
                        {
                            'point_detections': (new_detected_keypoints_l, new_detected_keypoints_r),
                            'robot_arm': robot_arm, 
                            'cam': cam, 
                            'cam_T_b': cam_T_b,
                            'joint_angle_readings': new_joint_angles,
                            'gamma': 0.15, 
                        },
                        
                        #shaftFeatureObs_kornia arguments
                        {
                            'use_lines': 'canny',
                            'use_clouds': None,
                            'detected_lines': {
                                'canny': (new_canny_lines_l, new_canny_lines_r),
                                'detected_endpoint_lines': (new_detected_endpoint_lines_l, new_detected_endpoint_lines_r),
                                'endpoint_cloud_lines': (new_endpoint_cloud_lines_l, new_endpoint_cloud_lines_r),
                                'line_cloud_lines': (new_line_cloud_lines_l, new_line_cloud_lines_r)
                            },
                            'intensity_clouds': {
                                'endpoint_clouds': (new_endpoint_clouds_l, new_endpoint_clouds_r),
                                'line_clouds': (new_line_clouds_l, new_line_clouds_r)
                            },
                            'robot_arm': robot_arm, 
                            'cam': cam, 
                            'cam_T_b': cam_T_b,
                            'joint_angle_readings': new_joint_angles,
                            'cost_assoc_params': {
                                'gamma_rho': 1, 
                                'gamma_theta': 1, 
                                'rho_thresh': 2,
                                'theta_thresh': 2
                            },
                            'pixel_probability_params': {
                                'sigma2_x': 0.5,
                                'sigma2_y': 0.5,
                            }
                        }
            ]

            pf.updateStep(upd_args)
            prev_joint_angles = new_joint_angles

            correction_estimation = pf.getMeanParticle()

            rospy.loginfo("Time to predict & update {}".format(time.time() - start_t))

            # Project and draw skeleton
            T = poseToMatrix(correction_estimation[:6])  
            new_joint_angles[-(correction_estimation.shape[0]-6):] += correction_estimation[6:]
            robot_arm.updateJointAngles(new_joint_angles)
            img_list = projectSkeleton(robot_arm.getSkeletonPoints(), np.dot(cam_T_b, T), [new_left_img, new_right_img], cam.projectPoints)
            img_list = drawShaftLines(robot_arm.getShaftFeatures(), cam, np.dot(cam_T_b, T), img_list)
            cv2.imshow("Left Img",  img_list[0])
            cv2.imshow("Right Img", img_list[1])
            cv2.waitKey(1)
            
        
        rate.sleep()