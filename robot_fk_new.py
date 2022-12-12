import numpy as np
import json
from utils import *

from fk_functions import compute_arm_fk,compute_wrist_fk

class Taurus_FK:
    def __init__(self, point_feature_json):
        
        # left base frame w.r.t taurus base frame
        self.T_b_lb = np.array([[-1.00000000e+00,  2.66960000e-06,  1.19284845e-09, 0.147],
                                [-2.66960000e-06, -1.00000000e+00,  8.65460159e-09, -0.13066],
                                [ 1.19287155e-09,  8.65459841e-09,  1.00000000e+00, -1.4272e-06],
                                [ 0.0       , 0.0        , 0.0        , 1.0     ]]) 
                                            
        with open(point_feature_json, 'r') as openfile:
            json_object = json.load(openfile)
            
        self.point_features = json_object
                                            
    def get_elbow_transfrom(self, theta0, theta1, theta2, theta3):
        ## this is J5
        joint0_pos, joint1_pos, joint2_pos, joint3_pos, joint_axes, final_transform = compute_arm_fk(theta0, theta1, theta2, theta3)
        return self.T_b_lb @ final_transform

    def get_end_frame_transform(self, theta0, theta1, theta2, theta3, theta4, theta5, theta6):
        ## this is J6
        joint0_pos, joint1_pos, joint2_pos, joint3_pos, joint_axes, final_transform = compute_arm_fk(theta0, theta1, theta2, theta3)
        pose = compute_wrist_fk(final_transform, theta4, theta5, theta6) #  take negative for the writst joints
        return self.T_b_lb @ pose

    def get_point_features(self, theta):
        T_b_J5 = self.get_elbow_transfrom(theta[0],theta[1],theta[2],theta[3])
        pitch_2_front = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['pitch_2:front']['position']).reshape(-1,1),1)).reshape(-1))
        pitch_2_back = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['pitch_2:back']['position']).reshape(-1,1),1)).reshape(-1))
        pitch_1_right = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['pitch_1:right']['position']).reshape(-1,1),1)).reshape(-1))
        pitch_1_left = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['pitch_1:left']['position']).reshape(-1,1),1)).reshape(-1))
        
        roll_1_front = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['roll_1:front']['position']).reshape(-1,1),1)).reshape(-1))
        roll_1_back = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['roll_1:back']['position']).reshape(-1,1),1)).reshape(-1))
        roll_2_right = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['roll_2:right']['position']).reshape(-1,1),1)).reshape(-1))
        roll_2_left = dehomogenize_3d(T_b_J5 @ np.vstack((np.array(self.point_features['roll_2:left']['position']).reshape(-1,1),1)).reshape(-1))

        T_b_J6 = self.get_end_frame_transform(theta[0],theta[1],theta[2],theta[3],\
                                                        theta[4],theta[5],theta[6])
        pitch_3_front = dehomogenize_3d(T_b_J6 @ np.vstack((np.array(self.point_features['pitch_3:front']['position']).reshape(-1,1),1)).reshape(-1))
        pitch_3_back = dehomogenize_3d(T_b_J6 @ np.vstack((np.array(self.point_features['pitch_3:back']['position']).reshape(-1,1),1)).reshape(-1))
        yaw_1 = dehomogenize_3d(T_b_J6 @ np.vstack((np.array(self.point_features['yaw_1']['position']).reshape(-1,1),1)).reshape(-1))
        yaw_2 = dehomogenize_3d(T_b_J6 @ np.vstack((np.array(self.point_features['yaw_2']['position']).reshape(-1,1),1)).reshape(-1))

        point_features = dict()
        point_features["roll_1:front"] = roll_1_front
        point_features["roll_1:back"] = roll_1_back
        point_features["roll_2:right"] = roll_2_right
        point_features["roll_2:left"] = roll_2_left
        point_features["pitch_2:front"] = pitch_2_front
        point_features["pitch_2:back"] = pitch_2_back
        point_features["pitch_1:right"] = pitch_1_right
        point_features["pitch_1:left"] = pitch_1_left
        point_features["pitch_3:front"] = pitch_3_front
        point_features["pitch_3:back"] = pitch_3_back
        point_features["yaw_1"] = yaw_1
        point_features["yaw_2"] = yaw_2

        return point_features



    

