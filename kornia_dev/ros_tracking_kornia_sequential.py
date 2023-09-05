import time
import rospy
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

# main function
if __name__ == "__main__":
    # Initalize ROS stuff here
    #rospy.init_node('robot_tool_tracking', anonymous=True)
    
    # reference image directory
    source_dir = 'kornia_dev/fei_ref_data/'
    draw_contours = False

    # annotate output with detected lines
    draw_lines = True

    # crop parameters
    in_file = source_dir + 'crop_scale.npy'
    crop_scale = np.load(in_file)
    print('crop_scale: {}'.format(crop_scale))
 
    # ref line indices
    in_file = source_dir + 'crop_ref_lines_l_idx.npy'
    crop_ref_lines_l_idx = np.load(in_file) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]
    
    in_file = source_dir + 'crop_ref_lines_r_idx.npy'
    crop_ref_lines_r_idx = np.load(in_file) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]

    # ref lines
    in_file = source_dir + 'crop_ref_lines_l.npy'
    crop_ref_lines_l = np.load(in_file)
    crop_ref_lines_l = torch.tensor(crop_ref_lines_l)

    in_file = source_dir + 'crop_ref_lines_r.npy'
    crop_ref_lines_r = np.load(in_file)
    crop_ref_lines_r = torch.tensor(crop_ref_lines_r)
    
    # line descriptors
    in_file = source_dir + 'crop_ref_desc_l.npy'
    crop_ref_desc_l = np.load(in_file)
    crop_ref_desc_l = torch.tensor(crop_ref_desc_l)

    in_file = source_dir + 'crop_ref_desc_r.npy'
    crop_ref_desc_r = np.load(in_file)
    crop_ref_desc_r = torch.tensor(crop_ref_desc_r)
    
    # reference images
    # left camera
    crop_ref_l_img = source_dir + 'ref_left_img.jpg'
    crop_ref_l_img = cv2.imread(crop_ref_l_img, cv2.IMREAD_COLOR)
    crop_ref_l_img = cv2.cvtColor(crop_ref_l_img, cv2.COLOR_BGR2RGB)
    img_dims = (int(crop_ref_l_img.shape[1]), int(crop_ref_l_img.shape[0]))
    crop_ref_l_tensor = K.image_to_tensor(crop_ref_l_img).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
    crop_ref_l_tensor = K.enhance.sharpness(crop_ref_l_tensor, 5.0)
    crop_ref_l_tensor = K.enhance.adjust_saturation(crop_ref_l_tensor, 5.0)
    crop_ref_l_tensor = K.color.rgb_to_grayscale(crop_ref_l_tensor) # [0, 1] torch.Size([1, 720, 1080]) torch.float32

    # right camera
    crop_ref_r_img = source_dir + 'ref_right_img.jpg'
    crop_ref_r_img = cv2.imread(crop_ref_r_img, cv2.IMREAD_COLOR)
    crop_ref_r_img = cv2.cvtColor(crop_ref_r_img, cv2.COLOR_BGR2RGB)
    crop_ref_r_tensor = K.image_to_tensor(crop_ref_r_img).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
    crop_ref_r_tensor = K.enhance.sharpness(crop_ref_r_tensor, 5.0)
    crop_ref_r_tensor = K.enhance.adjust_saturation(crop_ref_r_tensor, 5.0)
    crop_ref_r_tensor = K.color.rgb_to_grayscale(crop_ref_r_tensor) # [0, 1] torch.Size([1, 720, 1080]) torch.float32

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
        'use_endpoint_intensities_only': True,
        'endpoint_intensities_to_polar': False,
        'search_radius': 10.0,
        'intensity_params': {
            'use_metric': 'pct',
            'mean': 0,
            'std': 1.0,
            'pct': 10.0
        },
        'ransac_params': {
            'num_iterations': 5,
            'min_samples': 3.0,
            'residual_threshold': 0.75,
            'max_trials': 100,
            'img_dims': img_dims
        },
        'use_line_intensities_only': False,
        'line_intensities_to_polar': False
    } 

    # video recording
    record_video = True
    fps = 30
    if (record_video):

        #out_file = source_dir + 'canny_left_video.mp4'
        #out_file = source_dir + 'endp2p_left_video.mp4'
        #out_file = source_dir + 'endpi_left_video.mp4'
        #out_file = source_dir + 'endpi2p_left_video.mp4'
        #out_file = source_dir + 'li_left_video.mp4'
        #out_file = source_dir + 'li2p_left_video.mp4'
        #left_video_out  = cv2.VideoWriter(out_file,  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_dims)

        #out_file = source_dir + 'canny_right_video.mp4'
        #out_file = source_dir + 'endp2p_right_video.mp4'
        out_file = source_dir + 'endpi_right_video.mp4'
        #out_file = source_dir + 'endpi2p_right_video.mp4'
        #out_file = source_dir + 'li_right_video.mp4'
        #out_file = source_dir + 'li2p_right_video.mp4'
        right_video_out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_dims)

    # evaluation recording
    #accuracy_file = None
    #accuracy_file = open('kornia_dev/fei_ref_data/canny_accuracy.txt', 'w')
    #accuracy_file = open('kornia_dev/fei_ref_data/endpoints_to_polar_accuracy.txt', 'w')
    accuracy_file = open('kornia_dev/fei_ref_data/endpoint_intensities_only_accuracy.txt', 'w')
    #accuracy_file = open('kornia_dev/fei_ref_data/endpoint_intensities_to_polar_accuracy.txt', 'w')
    #accuracy_file = open('kornia_dev/fei_ref_data/line_intensities_only_accuracy.txt', 'w')
    #accuracy_file = open('kornia_dev/fei_ref_data/line_intensities_to_polar_accuracy.txt', 'w')

    #localization_file = None
    #localization_file = open('kornia_dev/fei_ref_data/canny_localization.txt', 'w')
    #localization_file = open('kornia_dev/fei_ref_data/endpoints_to_polar_localization.txt', 'w')
    localization_file = open('kornia_dev/fei_ref_data/endpoint_intensities_only_localization.txt', 'w')
    #localization_file = open('kornia_dev/fei_ref_data/endpoint_intensities_to_polar_localization.txt', 'w')
    #localization_file = open('kornia_dev/fei_ref_data/line_intensities_only_localization.txt', 'w')
    #localization_file = open('kornia_dev/fei_ref_data/line_intensities_to_polar_localization.txt', 'w')

    robot_arm = RobotLink(robot_file, use_dh_offset=False) # position / orientation in Meters
    cam = StereoCamera(camera_file, rectify = True, crop_scale = crop_scale, downscale_factor = 2, scale_baseline=1e-3)

    # Load hand-eye transform 
    # originally in M
    f = open(hand_eye_file)
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM2_tvec'])/1000.0 # convert to mm
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM2_rvec'])

    # Initialize filter
    pf = ParticleFilter(num_states=6, # originally 9 (6 for lumped error + 3 for endowrist pitch/yaw/squeeze) -> 6 for just lumped error
                        initialDistributionFunc=sampleNormalDistribution,
                        motionModelFunc=additiveGaussianNoise,
                        #motionModelFunc=lumpedErrorMotionModel,
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
                                    #0.0, 0.0, 0.0
                                    ])   # joints
                  }

    pf.initializeFilter(**init_kwargs)
    
    #rospy.loginfo("Initialized particle filter")
       
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

    msg_counter = 1

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
        
        #if (msg_counter < 8338):
            #msg_counter += 1
            #continue
        start_t = time.time()

        # copy l/r images so not overwritten by callback
        new_left_img = _cb_left_img.copy()
        new_right_img = _cb_right_img.copy()

        # process callback images
        new_left_img, new_right_img = cam.processImage(new_left_img, new_right_img, crop_scale = crop_scale)
        non_annotated_left_img = new_left_img.copy()
        non_annotated_right_img = new_right_img.copy()

        detected_keypoints_l, annotated_left_img  = segmentColorAndGetKeyPoints(non_annotated_left_img,  draw_contours = draw_contours)
        new_detected_keypoints_l = np.copy(detected_keypoints_l)
        detected_keypoints_r, annotated_right_img = segmentColorAndGetKeyPoints(non_annotated_right_img, draw_contours = draw_contours)
        new_detected_keypoints_r = np.copy(detected_keypoints_r)

        output_l  = detectShaftLines(
                                    non_annotated_img = non_annotated_left_img,
                                    annotated_img = annotated_left_img,
                                    ref_img = crop_ref_l_img,
                                    ref_tensor = crop_ref_l_tensor,
                                    crop_ref_lines = crop_ref_lines_l,
                                    crop_ref_lines_idx = crop_ref_lines_l_idx,
                                    crop_ref_desc = crop_ref_desc_l,
                                    model = model,
                                    draw_lines = draw_lines,
                                    canny_params = canny_params,
                                    kornia_params = kornia_params
                                    )
        output_r  = detectShaftLines(
                                    non_annotated_img = non_annotated_right_img,
                                    annotated_img = annotated_right_img,
                                    ref_img = crop_ref_r_img,
                                    ref_tensor = crop_ref_r_tensor,
                                    crop_ref_lines = crop_ref_lines_r,
                                    crop_ref_lines_idx = crop_ref_lines_r_idx,
                                    crop_ref_desc = crop_ref_desc_r,
                                    model = model,
                                    draw_lines = draw_lines,
                                    canny_params = canny_params,
                                    kornia_params = kornia_params
                                    )


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
        #j_change = new_joint_angles - prev_joint_angles

        #std_j = np.abs(j_change)*0.01
        #std_j[-3:] = 0.0

        # pred_kwargs = {
        #                 "std_pos": 2.5e-5, # in Meters
        #                 "std_ori": 1.0e-4,
        #                 "robot_arm": robot_arm, 
        #                 "std_j": std_j,
        #                 "nb": 4
        #               }

        pred_kwargs = {
                        "std": [2.5e-5, 2.5e-5, 2.5e-5,  # in Meters THESE ARE THE MAIN TUNING PARAMETERS!
                                1.0e-4, 1.0e-4, 1.0e-4] # in radians
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
                        'gamma': 0.5, # THIS IS A MAIN TUNING PARAMETER FOR FILTER PERFORMANCE https://github.com/ucsdarclab/dvrk_particle_filter/blob/master/config/ex_vivo_dataset_configure_filter.json
                    },
                    
                    #shaftFeatureObs_kornia arguments
                    {
                        'use_lines': False,
                        'use_clouds': 'endpoint_clouds',
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
                            'gamma_rho': 0.05,  # THIS IS A MAIN TUNING PARAMETER FOR FILTER PERFORMANCE https://github.com/ucsdarclab/dvrk_particle_filter/blob/master/config/ex_vivo_dataset_configure_filter.json
                            'gamma_theta': 7.5, # THIS IS A MAIN TUNING PARAMETER FOR FILTER PERFORMANCE https://github.com/ucsdarclab/dvrk_particle_filter/blob/master/config/ex_vivo_dataset_configure_filter.json
                            'rho_thresh': 75,
                            'theta_thresh': 0.5
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

        #rospy.loginfo("Time to predict & update {}".format(time.time() - start_t))

        # Project and draw skeleton
        T = poseToMatrix(correction_estimation[:6])  
        if correction_estimation.shape[0] > 6:
            new_joint_angles[-(correction_estimation.shape[0]-6):] += correction_estimation[6:]
        robot_arm.updateJointAngles(new_joint_angles)
        img_list = projectSkeleton(robot_arm.getSkeletonPoints(), np.dot(cam_T_b, T), [new_left_img, new_right_img], cam.projectPoints, (new_detected_keypoints_l, new_detected_keypoints_r), accuracy_file, t, msg_counter)
        img_list = drawShaftLines(robot_arm.getShaftFeatures(), cam, np.dot(cam_T_b, T), img_list)
        #print('ros_tracking_kornia_sequential.py np.dot(cTb, T): {}'.format(np.dot(cam_T_b, T)))
        #cv2.imshow("Left Img",  img_list[0])
        #cv2.imshow("Right Img", img_list[1])

        # video recording
        if (record_video):
            #print('img_list[0].shape: {}'.format(img_list[0].shape))
            #print('type(img_list[0]): {}'.format(type(img_list[0])))
            #left_video_out.write(img_list[0])
            right_video_out.write(img_list[1])

        text_string = str(t) + ',' + str(msg_counter) + ',' + str(new_joint_angles) + ',' + str(np.dot(cam_T_b, T)) + '\n'
        print(text_string)
        if (localization_file):
            localization_file.write(text_string)
        
        msg_counter += 1
        print('msg_counter: {}'.format(msg_counter))
        #cv2.waitKey(1)

    if (accuracy_file):
        accuracy_file.close()
    if (localization_file):
        localization_file.close()
    print('end of bag, closing bag')
    bag.close()
    print('total number of messages: {}'.format(msg_counter))
    print('Releasing video capture')
    if (record_video):
        #left_video_out.release()
        right_video_out.release()