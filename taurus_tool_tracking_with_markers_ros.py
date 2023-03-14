import sys
import os

import numpy as np
import time
import cv2
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
from robot_fk_new import *
from camera import *
from utils import *
from particle_filter import *
from fk_functions import *
import imutils
from scipy import optimize
from scipy.stats import norm

import matplotlib.pyplot as plt
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

import sensor_msgs
import geometry_msgs

from PIL import Image as PILImage

bridge = CvBridge()

import yaml

############### read config file ##################
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


########################### essential functions ####################################

def segmentColorAndGetKeyPoints(img, hsv_min=(90, 40, 40), hsv_max=(120, 255, 255), draw_contours=False):
    hsv = cv2.cvtColor(img,  cv2.COLOR_RGB2HSV)
    mask  = cv2.inRange(hsv , hsv_min, hsv_max)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    centroids = []
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = M["m10"]
            cY = M["m01"]
        else:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        centroids.append(np.array([cX, cY]))

    if draw_contours:
        cv2.drawContours(img, cnts, -1, (255, 0, 0), thickness=3)
    
    return np.array(centroids), img

#kwargs must have init_rvec, init_tvec, and joint_angles
def project_feature_points(x, **kwargs):
    

    theta = kwargs["joint_angles"]
    robot = kwargs["robot"]
    cam = kwargs["camera"]
    point_features = taurus.get_point_features(theta)

    f0 = point_features['roll_1:front']
    f1 = point_features['roll_1:back']
    f2 = point_features['roll_2:right']
    f3 = point_features['roll_2:left']
    f4 = point_features['pitch_1:right']
    f5 = point_features['pitch_1:left']
    f6 = point_features['pitch_2:front']
    f7 = point_features['pitch_2:back']
    ### markers on the wrist joint
    f8 = point_features['pitch_3:front']
    f9 = point_features['pitch_3:back']
    ### markers on gripper
    f10 = point_features['yaw_1']
    f11 = point_features['yaw_2']



    n_points = len(point_features.keys())
    y = np.zeros((x.shape[0], n_points, 2))



    for i, particle in enumerate(x):
        
        rvec, tvec = cv2.composeRT( kwargs["init_rvec"], kwargs["init_tvec"], particle[3:], particle[:3])[:2]
        
        p0,_ = cv2.projectPoints(f0, rvec, tvec, cam.P,cam.D)
        p0 = np.squeeze(p0) - cam.offset
        y[i, 0] = p0
        p1,_ = cv2.projectPoints(f1, rvec, tvec, cam.P,cam.D)
        p1 = np.squeeze(p1) - cam.offset
        y[i, 1] = p1
        p2,_ = cv2.projectPoints(f2, rvec, tvec, cam.P,cam.D)
        p2 = np.squeeze(p2) - cam.offset
        y[i, 2] = p2
        p3,_ = cv2.projectPoints(f3, rvec, tvec, cam.P,cam.D)
        p3 = np.squeeze(p3) - cam.offset
        y[i, 3] = p3
        p4,_ = cv2.projectPoints(f4, rvec, tvec, cam.P,cam.D)
        p4 = np.squeeze(p4) - cam.offset
        y[i, 4] = p4
        p5,_ = cv2.projectPoints(f5, rvec, tvec, cam.P,cam.D)
        p5 = np.squeeze(p5) - cam.offset
        y[i, 5] = p5
        p6,_ = cv2.projectPoints(f6, rvec, tvec, cam.P,cam.D)
        p6 = np.squeeze(p6) - cam.offset
        y[i, 6] = p6
        p7,_ = cv2.projectPoints(f7, rvec, tvec, cam.P,cam.D)
        p7 = np.squeeze(p7) - cam.offset
        y[i, 7] = p7
        ### markers on the wrist joint
        p8,_ = cv2.projectPoints(f8, rvec, tvec, cam.P,cam.D)
        p8 = np.squeeze(p8) - cam.offset
        y[i, 8] = p8
        p9,_ = cv2.projectPoints(f9, rvec, tvec, cam.P,cam.D)
        p9 = np.squeeze(p9) - cam.offset
        y[i, 9] = p9
        ### markers on the gripper
        p10,_ = cv2.projectPoints(f10, rvec, tvec, cam.P,cam.D)
        p10 = np.squeeze(p10) - cam.offset
        y[i, 10] = p10
        p11,_ = cv2.projectPoints(f11, rvec, tvec, cam.P,cam.D)
        p11 = np.squeeze(p11) - cam.offset
        y[i, 11] = p11
        
    return y


# Note that hypotheses is of dimensions N (number of particles) by P (number of features) by 2 
# and observed is of dimensions P by 2

def weight_function(hypotheses, observed, **kwargs):
    n_particles = hypotheses.shape[0]
    weights = np.zeros(n_particles)

    for i in range(n_particles):
        projected_points = hypotheses[i]
        # Use hungarian algorithm to match projected and detected points
        C = np.linalg.norm(projected_points[:, None, :] - observed[None, :,  :], axis=2)
        #print(C)
        row_idx, col_idx = optimize.linear_sum_assignment(C)

        # Use threshold to remove outliers
        idx_to_keep = C[row_idx, col_idx] < kwargs["association_threshold"]
        row_idx = row_idx[idx_to_keep]
        col_idx = col_idx[idx_to_keep]

        # Compute observation probability
        prob = np.mean(np.exp(-kwargs["gamma"]*C[row_idx, col_idx])) 
        #print(prob)
        weights[i] = prob

    weights[np.isnan(weights)] = 0        

    return weights

################################# defining parameters and initialize particle filter ############################3
hsv_min = np.array(cfg["hsv_min"])
hsv_max = np.array(cfg["hsv_max"])

# particle filter parameters
sigma_t = cfg["sigma_t"] 
sigma_r = cfg["sigma_r"]  
scale_init_sigma = cfg["scale_init_sigma"]
gamma = cfg["gamma"]
association_threshold=cfg["association_threshold"]
n_particles = cfg["n_particles"]
n_eff = cfg["n_eff"]
resample_proportion = cfg["resample_proportion"]

taurus = Taurus_FK("point_feature_markers.json")

T_b_c = np.array(cfg["initial_camera_to_base"])
T_c_b = np.linalg.inv(T_b_c)

rvec_init = cv2.Rodrigues(T_c_b[:3,:3])[0].squeeze()
tvec_init = T_c_b[:3,-1]

# initialize particle filter
pf = ParticleFilter(prior_fn=lambda n : prior_fn(n, scale_init_sigma, sigma_t, sigma_r), 
                    observe_fn=project_feature_points,
                    n_particles=n_particles,
                    dynamics_fn=identity,
                    noise_fn=lambda x, **kwargs: 
                                gaussian_noise(x, sigmas=[sigma_t, sigma_t, sigma_t, 
                                                          sigma_r, sigma_r, sigma_r]),
                    weight_fn=weight_function,
                    resample_proportion=resample_proportion,
                    resample_fn = stratified_resample,
                    n_eff_threshold = n_eff)
pf.init_filter() 

############################# main loop ################################################3

visualize = False
def gotData(img_msg, joint_msg):
    #print("Received data!")

    # receive images
    try:
        # Convert your ROS Image message to OpenCV2
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        img = cv_img[:,:,:3].copy()
        
    except CvBridgeError as e:
        print(e)

    # receive joint angles
    left_joint_angles_orderred = np.array(joint_msg.position)[[2, 4, 3, 0, 6, 5, 7]]

    # detect point features
    centroids, img = segmentColorAndGetKeyPoints(img, hsv_min, hsv_max, draw_contours=visualize)
    if centroids.shape[0] == 0:
        print("No marker detected!!!")
        centroids = None

    
    # estimate with particle filter
    pf.update(centroids, init_rvec=rvec_init, init_tvec=tvec_init,\
                joint_angles=left_joint_angles_orderred, association_threshold=association_threshold,\
                gamma=gamma, robot=taurus, camera=camera)
    
    # most recent prediction of the robot base pose
    rvec_new, tvec_new = cv2.composeRT(rvec_init, tvec_init, pf.mean_state[3:], pf.mean_state[:3], 
                           camera.P, camera.D)[:2]

    # transformation matrix
    R,_ = cv2.Rodrigues(rvec_new)
    #quat_tmp = quaternions.mat2quat(R)
    #quat = [quat_tmp[1],quat_tmp[2],quat_tmp[3],quat_tmp[0]]

    c_T_b = np.hstack((R,tvec_new))
    c_T_b = np.vstack((c_T_b,[0,0,0,1]))

    T_J5 = taurus.get_elbow_transfrom(left_joint_angles_orderred[0],\
                                            left_joint_angles_orderred[1],\
                                                left_joint_angles_orderred[2],\
                                                    left_joint_angles_orderred[3])

    T_J6 = taurus.get_end_frame_transform(left_joint_angles_orderred[0],\
                                            left_joint_angles_orderred[1],\
                                                left_joint_angles_orderred[2],\
                                                    left_joint_angles_orderred[3],\
                                                        left_joint_angles_orderred[4],\
                                                            left_joint_angles_orderred[5],\
                                                                left_joint_angles_orderred[6])

    print(img_msg.header.stamp)
    c_T_elbow = c_T_b @ T_J5
    c_T_wrist = c_T_b @ T_J6
    print(c_T_elbow)
    print(c_T_wrist)


    ####################################
    # draw estimation
    if visualize:
        point_features = taurus.get_point_features(left_joint_angles_orderred)
        image = img.copy()

        ori = c_T_elbow @ np.array([0,0,0,1])
        x = c_T_elbow @ np.array([0.02,0,0,1])
        y = c_T_elbow @ np.array([0,0.02,0,1])
        z = c_T_elbow @ np.array([0,0,0.02,1])

        px,_ = cv2.projectPoints(x[:3], np.zeros(3), np.zeros(3), P, D)
        py,_ = cv2.projectPoints(y[:3], np.zeros(3), np.zeros(3), P, D)
        pz,_ = cv2.projectPoints(z[:3], np.zeros(3), np.zeros(3), P, D)
        po,_ = cv2.projectPoints(ori[:3], np.zeros(3), np.zeros(3), P, D)

        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(px.squeeze().astype(int)), 
                                        (255,0,0), 5)
        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(py.squeeze().astype(int)), 
                                        (0,255,0), 5)
        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(pz.squeeze().astype(int)), 
                                        (0,0,255), 5)
        
            

        plt.imsave("test/" + str(time.time()) + ".png",image)
        


#############################################################################
rospy.init_node('taurus_tool_tracking')
# Define your image topic
image_topic = "/rgb/image_raw"
robot_joint_topic = "/joint_states"
#robot_pose_topic = "/robot_base_pose"

#### get camera info
camera_info = rospy.wait_for_message("/rgb/camera_info", sensor_msgs.msg.CameraInfo)
P = np.array(camera_info.P).reshape(3,4)[:,:3]
D = np.array(camera_info.D)
offset = np.array([camera_info.roi.x_offset,camera_info.roi.y_offset])
camera = Camera(P,D,offset)


# Set up  subscriber and define its callback
image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
#pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
ats.registerCallback(gotData)


# Main loop:
rate = rospy.Rate(30) # 30hz

while not rospy.is_shutdown():
    rate.sleep()