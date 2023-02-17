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
hsv_min=(120, 40, 40)
hsv_max=(150, 255, 255)

# particle filter parameters
sigma_t = 0.002 #0.005 #m
sigma_r = 0.001   #0.035 #2 degrees
#sigma_j = 0.001
#u_j     = 0.002
scale_init_sigma = 10
gamma = 0.5 # 1
association_threshold=500
n_particles = 500
n_eff = 0.5
resample_proportion = 0

taurus = Taurus_FK("point_feature_markers.json")

T_b_c = np.array([[-0.0086596 ,  0.86086372, -0.50876189, 0.667792],
                  [-0.9999511 , -0.00502532,  0.00851687, -0.027447],
                  [ 0.00477517,  0.50881076,  0.86086515, -0.0123208],
                  [ 0.0       , 0.0        , 0.0        , 1.0     ]])
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

visualize = True
start = time.time()
def gotData(img_msg, joint_msg):
    global start
    print("Received data!")

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
    centroids, img = segmentColorAndGetKeyPoints(img, hsv_min, hsv_max, draw_contours=True)

    
    # estimate with particle filter
    # estimate with particle filter
    pf.update(centroids, init_rvec=rvec_init, init_tvec=tvec_init,\
                joint_angles=left_joint_angles_orderred, association_threshold=association_threshold,\
                gamma=gamma, robot=taurus, camera=camera)
    
    # most recent prediction of the robot pose
    rvec_new, tvec_new = cv2.composeRT(rvec_init, tvec_init, pf.mean_state[3:], pf.mean_state[:3], 
                           camera.P, camera.D)[:2]


    ####################################
    # draw estimation
    if visualize:
        point_features = taurus.get_point_features(left_joint_angles_orderred)
        image = img.copy()

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
        ori = T_J5 @ np.array([0,0,0,1])
        x = T_J5 @ np.array([0.02,0,0,1])
        y = T_J5 @ np.array([0,0.02,0,1])
        z = T_J5 @ np.array([0,0,0.02,1])

        px,_ = cv2.projectPoints(x[:3], rvec_new, tvec_new, P, D)
        py,_ = cv2.projectPoints(y[:3], rvec_new, tvec_new, P, D)
        pz,_ = cv2.projectPoints(z[:3], rvec_new, tvec_new, P, D)
        po,_ = cv2.projectPoints(ori[:3], rvec_new, tvec_new, P, D)

        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(px.squeeze().astype(int)), 
                                        (255,0,0), 5)
        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(py.squeeze().astype(int)), 
                                        (0,255,0), 5)
        image = cv2.line(image, tuple(po.squeeze().astype(int)), tuple(pz.squeeze().astype(int)), 
                                        (0,0,255), 5)
        
            

        plt.imsave("test/" + str(time.time()) + ".png",image)
    ####
        


#############################################################################
rospy.init_node('taurus_tool_tracking')
# Define your image topic
image_topic = "/rgb/image_raw"
robot_joint_topic = "/joint_states"
robot_pose_topic = "robot_pose"
# Set up your subscriber and define its callback
#rospy.Subscriber(image_topic, sensor_msgs.msg.Image, gotData)

image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

camera_info = rospy.wait_for_message("/rgb/camera_info", sensor_msgs.msg.CameraInfo)
P = np.array(camera_info.P).reshape(3,4)[:,:3]
D = np.array(camera_info.D)
offset = np.array([camera_info.roi.x_offset,camera_info.roi.y_offset])
camera = Camera(P,D,offset)


ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
ats.registerCallback(gotData)


# Main loop:
rate = rospy.Rate(30) # 30hz

while not rospy.is_shutdown():
    rate.sleep()