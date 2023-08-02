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
    #print('crop_scale: {}'.format(crop_scale))

    # reference lines
    in_file = source_dir + 'crop_ref_lines_l.npy'
    crop_ref_lines_l = torch.as_tensor(np.load(in_file)) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]
    in_file = source_dir + 'crop_ref_lines_r.npy'
    crop_ref_lines_r = torch.as_tensor(np.load(in_file)) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]

    # sorted reference lines
    in_file = source_dir + 'crop_ref_lines_l_sorted.npy'
    crop_ref_lines_l_sorted = torch.as_tensor(np.load(in_file)) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]
    in_file = source_dir + 'crop_ref_lines_r_sorted.npy'
    crop_ref_lines_r_sorted = torch.as_tensor(np.load(in_file))
    
    # ref line indices
    in_file = source_dir + 'crop_ref_lines_l_idx.npy'
    crop_ref_lines_l_idx = np.load(in_file) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]
    in_file = source_dir + 'crop_ref_lines_r_idx.npy'
    crop_ref_lines_r_idx = np.load(in_file) # torch.Size([2, 2, 2]) # endpoints per line: [y, x] [y, x]

    # selected ref lines
    in_file = source_dir + 'crop_ref_lines_l_selected.npy'
    crop_ref_lines_l_selected = np.load(in_file)
    in_file = source_dir + 'crop_ref_lines_r_selected.npy'
    crop_ref_lines_r_selected = np.load(in_file)

    # reference images
    # left camera
    in_file = source_dir + 'crop_ref_l_img.npy'
    crop_ref_l_img = np.load(in_file) # (404, 720, 3) RGB uint8
    img_dims = (int(crop_ref_l_img.shape[1]), int(crop_ref_l_img.shape[0]))
    crop_ref_l_tensor = K.image_to_tensor(crop_ref_l_img).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
    crop_ref_l_tensor = K.color.rgb_to_grayscale(crop_ref_l_tensor) # [0, 1] torch.Size([1, 720, 1080]) torch.float32
    # right camera
    in_file = source_dir + 'crop_ref_r_img.npy'
    crop_ref_r_img = np.load(in_file) # (404, 720, 3) RGB uint8
    #print('crop_ref_r_img.shape: {}'.format(crop_ref_r_img.shape))
    crop_ref_r_tensor = K.image_to_tensor(crop_ref_r_img).float() / 255.0 # [0, 1] torch.Size([3, 720, 1080]) torch.float32
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
            'num_iterations': 5,
            'min_samples': 3.0,
            'residual_threshold': 0.75,
            'max_trials': 100,
            'img_dims': img_dims
        },
        'use_line_intensities_only': False,
        'line_intensities_to_polar': True
    } 

    # output directory for recordings
    video_out_dir = source_dir + 'video_recordings/'
    particle_out_dir = source_dir + 'particle_data/'
    if (canny_params['use_canny']):
        video_out_dir += 'use_canny/'
        particle_out_dir += 'use_canny/'
    elif (kornia_params['use_kornia'] and kornia_params['endpoints_to_polar']):
        video_out_dir += 'endpoints_to_polar/'
        particle_out_dir += 'endpoints_to_polar/'
    elif (kornia_params['use_kornia'] and kornia_params['use_endpoint_intensities_only']):
        video_out_dir += 'use_endpoint_intensities_only/'
        particle_out_dir += 'use_endpoint_intensities_only/'
    elif (kornia_params['use_kornia'] and kornia_params['endpoint_intensities_to_polar']):
        video_out_dir += 'endpoint_intensities_to_polar/'
        particle_out_dir += 'endpoint_intensities_to_polar/'
    elif (kornia_params['use_kornia'] and kornia_params['use_line_intensities_only']):
        video_out_dir += 'use_line_intensities_only/'
        particle_out_dir += 'use_line_intensities_only/'
    elif (kornia_params['use_kornia'] and kornia_params['line_intensities_to_polar']):
        video_out_dir += 'line_intensities_to_polar/'
        particle_out_dir += 'line_intensities_to_polar/'

    # video recording
    record_video = False
    fps = 30
    if (record_video):
        out_file = video_out_dir + 'left_video.mp4'
        left_video_out  = cv2.VideoWriter(out_file,  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_dims)
        out_file = video_out_dir + 'right_video.mp4'
        right_video_out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_dims)

    # particle recording
    record_particles = False
    record_particles_counter = 1

    robot_arm = RobotLink(robot_file, use_dh_offset=False) # position / orientation in Meters
    cam = StereoCamera(camera_file, rectify = True, crop_scale = crop_scale, downscale_factor = 2, scale_baseline=1e-3)

    # Load hand-eye transform 
    # originally in M
    f = open(hand_eye_file)
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0 # convert to mm
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])


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
    
    rospy.loginfo("Initialized particle filter")
       
    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_joint_angles = None

    try: 
        while not rospy.is_shutdown():
            if new_cb_data:
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

                '''
                annotated_left_ref_img, annotated_left_img = makeShaftAssociations(
                                            new_img = annotated_left_img, 
                                            ref_tensor = crop_ref_l_tensor,
                                            ref_img = crop_ref_l_img,
                                            crop_ref_lines = crop_ref_lines_l,
                                            crop_ref_lines_sorted = crop_ref_lines_l_sorted,
                                            crop_ref_lines_selected = crop_ref_lines_l_selected,
                                            crop_ref_lines_idx = crop_ref_lines_l_idx,
                                            model = model
                                            )
                cv2.imshow('ref_img_l', ref_img_l)
                cv2.imshow('new_left_img', new_left_img)
                
                annotated_right_ref_img, annotated_right_img = makeShaftAssociations(
                                            new_img = annotated_right_img, 
                                            ref_tensor = crop_ref_r_tensor,
                                            ref_img = crop_ref_r_img,
                                            crop_ref_lines = crop_ref_lines_r,
                                            crop_ref_lines_sorted = crop_ref_lines_r_sorted,
                                            crop_ref_lines_selected = crop_ref_lines_r_selected,
                                            crop_ref_lines_idx = crop_ref_lines_r_idx,
                                            model = model
                                            )
                cv2.imshow('ref_img_r', ref_img_r)
                cv2.imshow('new_right_img', new_right_img)
                '''

                output_l  = detectShaftLines(
                                            non_annotated_img = non_annotated_left_img,
                                            annotated_img = annotated_left_img,
                                            ref_img = crop_ref_l_img,
                                            ref_tensor = crop_ref_l_tensor,
                                            crop_ref_lines = crop_ref_lines_l,
                                            crop_ref_lines_idx = crop_ref_lines_l_idx,
                                            crop_ref_lines_selected = crop_ref_lines_l_selected,
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
                                            crop_ref_lines_selected = crop_ref_lines_r_selected,
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

                rospy.loginfo("Time to predict & update {}".format(time.time() - start_t))

                # Project and draw skeleton
                T = poseToMatrix(correction_estimation[:6])  
                if correction_estimation.shape[0] > 6:
                    new_joint_angles[-(correction_estimation.shape[0]-6):] += correction_estimation[6:]
                robot_arm.updateJointAngles(new_joint_angles)
                img_list = projectSkeleton(robot_arm.getSkeletonPoints(), np.dot(cam_T_b, T), [new_left_img, new_right_img], cam.projectPoints, (new_detected_keypoints_l, new_detected_keypoints_r))
                img_list = drawShaftLines(robot_arm.getShaftFeatures(), cam, np.dot(cam_T_b, T), img_list)
                cv2.imshow("Left Img",  img_list[0])
                cv2.imshow("Right Img", img_list[1])

                # video recording
                if (record_video):
                    #print('img_list[0].shape: {}'.format(img_list[0].shape))
                    #print('type(img_list[0]): {}'.format(type(img_list[0])))
                    left_video_out.write(img_list[0])
                    right_video_out.write(img_list[1])

                # particle recording
                if (record_particles):
                    out_file = particle_out_dir + str(record_particles_counter) + '.npy'
                    particles = pf._particles.copy()
                    #print('particles: {}'.format(particles))
                    #print('particles.shape: {}'.format(particles.shape))
                    weights = pf._weights.copy()
                    #print('weights: {}'.format(weights))
                    #print('weights.shape: {}'.format(weights.shape))
                    out_data = [particles, weights, np.dot(weights, particles)]
                    np.save(out_file, out_data)

                record_particles_counter += 1
                cv2.waitKey(1)
            else:
                rate.sleep()
    
    #except ValueError:
        #print('value error')
        #print(record_particles_counter)
        #pass

    except KeyboardInterrupt: 
        print('Broke rospy loop')
        print('Releasing video capture')
        if (record_video):
            left_video_out.release()
            right_video_out.release()