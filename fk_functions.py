#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:57:23 2022

@author: nmarion
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:05:10 2022

@author: SRI International
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import pi
   
def compute_arm_fk(theta0, theta1, theta2, theta3):
    
    joint_0 = np.eye(4)
    joint_0[0:3,0:3] = np.array([[np.cos(theta0), -np.sin(theta0),0],[np.sin(theta0),np.cos(theta0),0],[0,0,1]])
    
    joint_1 = np.eye(4)
    joint_1[0:3,0:3] = np.array([[np.cos(theta1), -np.sin(theta1),0],[np.sin(theta1),np.cos(theta1),0],[0,0,1]])
    
    joint_2 = np.eye(4)
    joint_2[0:3,0:3] = np.array([[np.cos(theta2),  -np.sin(theta2),0],[np.sin(theta2),np.cos(theta2),0],[0,0,1]])
    joint_2[:-1,-1] = [0,0,0.2112772]
    
    joint_3 = np.eye(4)
    joint_3[0:3,0:3] = np.array([[np.cos(theta3), -np.sin(theta3),0],[np.sin(theta3),np.cos(theta3),0],[0,0,1]])
    joint_3[:-1,-1] = [0.03175, 0, 0.0015875]
    
    tip_transform = np.eye(4)
    tip_transform[:-1,-1] = [0.3046984, -0.01999996,0]
        
    #Rotate about x -pi/2
    x_transform = np.zeros((4,4))
    x_transform[0,0] = 1
    x_transform[1,2] = 1
    x_transform[2,1] = -1
    x_transform[-1,-1] = 1
    
    #Rotate about x pi/2
    x_transform1 = np.zeros((4,4))
    x_transform1[0,0] = 1
    x_transform1[1,2] = -1
    x_transform1[2,1] = 1
    x_transform1[-1,-1] = 1
    
    #Rotate about y -pi/2
    y_transform = np.zeros((4,4))
    y_transform[0,2] = -1
    y_transform[1,1] = 1
    y_transform[2,0] = 1
    y_transform[-1,-1] = 1
    
     #Rotate about z -pi/2
    z_transform = np.zeros((4,4))
    z_transform[0,1] = 1
    z_transform[1,0] = -1
    z_transform[2,2] = 1
    z_transform[-1,-1] = 1
    

    transform0= (x_transform @ joint_0)
    joint0_pos = transform0[:-1,-1]
    axis0 = R.from_matrix(transform0[:3,:3]).as_euler('xyz')
    
    transform1= transform0 @ ( x_transform1 @ joint_1 )
    joint1_pos = transform1[:-1,-1]
    axis1 = R.from_matrix(transform1[:3,:3]).as_euler('xyz')
    
    transform2= transform1 @ (y_transform @ joint_2)
    joint2_pos = transform2[:-1,-1]
    axis2 = R.from_matrix(transform2[:3,:3]).as_euler('xyz')
    
    transform3= transform2 @ ( z_transform @ x_transform @ joint_3)
    joint3_pos = transform3[:-1,-1]
    axis3 = R.from_matrix(transform3[:3,:3]).as_euler('xyz')
        
    final_transform = transform3 @ tip_transform
    
    joint_axes = [axis0, axis1, axis2, axis3]
    
    return joint0_pos, joint1_pos, joint2_pos, joint3_pos, joint_axes, final_transform


def compute_wrist_fk (elbow_transform, wrist_roll_angle, wrist_pitch_angle, wrist_yaw_angle):
       
    wrist_roll = np.eye(4)
    wrist_roll[0:3,0:3] = np.array([[np.cos(wrist_roll_angle), -np.sin(wrist_roll_angle),0],[np.sin(wrist_roll_angle),np.cos(wrist_roll_angle),0],[0,0,1]])
    
    wrist_pitch = np.eye(4)
    wrist_pitch[0:3,0:3] = np.array([[np.cos(wrist_pitch_angle), -np.sin(wrist_pitch_angle),0],[np.sin(wrist_pitch_angle),np.cos(wrist_pitch_angle),0],[0,0,1]])
    
    wrist_yaw = np.eye(4)
    wrist_yaw[0:3,0:3] = np.array([[np.cos(wrist_yaw_angle),  -np.sin(wrist_yaw_angle),0],[np.sin(wrist_yaw_angle),np.cos(wrist_yaw_angle),0],[0,0,1]])
    wrist_yaw[:-1,-1] = [0,0,0.009017]
    
    #Rotate about x -pi/2
    x_transform = np.zeros((4,4))
    x_transform[0,0] = 1
    x_transform[1,2] = 1
    x_transform[2,1] = -1
    x_transform[-1,-1] = 1
    
    #Rotate about y pi/2
    y_transform = np.zeros((4,4))
    y_transform[0,2] = 1
    y_transform[1,1] = 1
    y_transform[2,0] = -1
    y_transform[-1,-1] = 1        

    #Rotate about y -pi/2
    y_transform1 = np.zeros((4,4))
    y_transform1[0,2] = -1
    y_transform1[1,1] = 1
    y_transform1[2,0] = 1
    y_transform1[-1,-1] = 1
    
    #Rotate about z pi/2
    z_transform = np.zeros((4,4))
    z_transform[0,1] = -1
    z_transform[1,0] = 1
    z_transform[2,2] = 1
    z_transform[-1,-1] = 1

    
    wrist_roll_output =  elbow_transform @ (y_transform @ wrist_roll)
    
    wrist_pitch_output = wrist_roll_output @ (z_transform @ y_transform1  @ wrist_pitch)
    
    final_output = wrist_pitch_output @ (x_transform @ wrist_yaw)

    
    return final_output
