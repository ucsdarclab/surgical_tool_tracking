import numpy as np
import json

class Taurus_FK:
    def __init__(self, point_feature_json):
        
        '''
        (   a(i-1)        alpha(i-1)       D(i)        Theta     Theta offset(i)  Range of motion )
        0.00000	   0.0	          0.00000	99999	    0.0 	+243.7	 -114.77  shoulder_pitch (joint/link 1)
        0.00000	  90.0	          0.00000	99999	   90.0 	  43.00	  -43.87  shoulder_yaw   (joint/link 2)
        0.00000	 -90.0	          8.31800	99999	  180.0 	+164.75	 -164.75  upper_arm_roll (joint/link 3)
        1.25000         -90.0	          0.06250	99999	  -90.0 	 +82.24	  -69.28  elbow          (joint/link 4)
        0.78740         -90.0            11.9960 	99999     -90.0		+360.00  -180.00  tool_roll      (joint/link 5)
        0.00000         -90.0            0.00000       99999     -90.0 	 +75.00	  -75.00  tool_pitch     (joint/link 6)	
        0.35500         -90.0            0.00000       99999       0.0 	 +75.00	  -75.00  tool_yaw       (joint/link 7)
        0.35500          90.0            0.00000       99999       0.0 	 +75.00	  -75.00  tool_jaw_yaw   (joint/link 8)
        '''
        self.mount_T_base_left = np.array([[-1.00000, 0.00000, 0.00000, 5.8122],
                                            [0.00000, 0.00000, -1.00000, -5.1250],
                                            [0.00000, -1.00000, 0.00000, 0.0000],
                                            [0.00000, 0.00000, 0.00000, 1.0000]])
        self.mount_T_base_right = np.array([[-1.00000, 0.00000, 0.00000, 5.8122],
                                            [0.00000, 0.00000, 1.00000, 5.1250],
                                            [0.00000, 1.00000, 0.00000, 0.0000],
                                            [0.00000, 0.00000, 0.00000, 1.0000]])
                                            
        with open(point_feature_json, 'r') as openfile:
            json_object = json.load(openfile)
            
        self.point_features = json_object
                                            
    
    def T_from_DH(self,a,alp,d,the):
        '''
        Transformation matrix fron DH
        '''
        T = np.array([[np.cos(the), -np.sin(the), 0, a],
                    [np.sin(the)*np.cos(alp), np.cos(the)*np.cos(alp), -np.sin(alp), -d*np.sin(alp)],
                    [np.sin(the)*np.sin(alp), np.cos(the)*np.sin(alp), np.cos(alp), d*np.cos(alp)],
                    [0,0,0,1]])
        return T

    def get_tip_pose_right(self,theta):
        
        T_0_1 = self.T_from_DH(0, 0, 0, theta[0])
        T_1_2 = self.T_from_DH(0, np.pi/2,0, theta[1] + np.pi/2)
        T_2_3 = self.T_from_DH(0, -np.pi/2, 8.318 , theta[2] + np.pi)
        T_3_4 = self.T_from_DH(1.25, -np.pi/2, 0.0625, theta[3] - np.pi/2)
        T_4_5 = self.T_from_DH(0.78740, -np.pi/2, 11.9960, theta[4] - np.pi/2)
        T_5_6 = self.T_from_DH(0, -np.pi/2, 0, theta[5] - np.pi/2)
        T_6_7 = self.T_from_DH(0.355, -np.pi/2, 0, theta[6])

        tip_pose = self.mount_T_base_right @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7

        return tip_pose
    
    def get_tip_pose_left(self,theta):
        
        T_0_1 = self.T_from_DH(0, 0, 0, theta[0])
        T_1_2 = self.T_from_DH(0, np.pi/2,0, theta[1] + np.pi/2)
        T_2_3 = self.T_from_DH(0, -np.pi/2, 8.318 , theta[2] + np.pi)
        T_3_4 = self.T_from_DH(1.25, -np.pi/2, 0.0625, theta[3] - np.pi/2)
        T_4_5 = self.T_from_DH(0.78740, -np.pi/2, 11.9960, theta[4] - np.pi/2)
        T_5_6 = self.T_from_DH(0, -np.pi/2, 0, theta[5] - np.pi/2)
        T_6_7 = self.T_from_DH(0.355, -np.pi/2, 0, theta[6])

        tip_pose = self.mount_T_base_left @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7

        return tip_pose
    
    def get_FK_tree(self,theta):
        
        FK_tree = dict()
        T_0_1 = self.T_from_DH(0, 0, 0, theta[0])
        FK_tree['J1'] = T_0_1
        T_1_2 = self.T_from_DH(0, np.pi/2,0, theta[1] + np.pi/2)
        FK_tree['J2'] = FK_tree['J1'] @ T_1_2
        T_2_3 = self.T_from_DH(0, -np.pi/2, 8.318 , theta[2] + np.pi)
        FK_tree['J3'] = FK_tree['J2'] @ T_2_3
        T_3_4 = self.T_from_DH(1.25, -np.pi/2, 0.0625, theta[3] - np.pi/2)
        FK_tree['J4'] = FK_tree['J3'] @ T_3_4
        T_4_5 = self.T_from_DH(0.78740, -np.pi/2, 11.9960, theta[4] - np.pi/2)
        FK_tree['J5'] = FK_tree['J4'] @ T_4_5
        T_5_6 = self.T_from_DH(0, -np.pi/2, 0, theta[5] - np.pi/2)
        FK_tree['J6'] = FK_tree['J5'] @ T_5_6
        T_6_7 = self.T_from_DH(0.355, -np.pi/2, 0, theta[6])
        FK_tree['J7'] = FK_tree['J6'] @ T_6_7

        return FK_tree
