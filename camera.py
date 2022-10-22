import numpy as np
from utils import *
import cv2

class Camera:
    def __init__(self, intrinsic, distortion, offset=None):
        self.P = intrinsic
        self.D = distortion
        self.offset = offset
        
    def draw_skeleton(self, theta, rvec, tvec, image, robot, color=(0,255,0), thickness=8):

        FK_tree = robot.get_FK_tree(theta)

        f0 = dehomogenize_3d(robot.mount_T_base_left @ FK_tree["J4"] \
                             @ np.vstack((np.array([0.0, 0.0, 0.0]).reshape(-1,1),1))).reshape(-1)
        p0,_ = cv2.projectPoints(f0, rvec, tvec, self.P,self.D)
        p0 = np.squeeze(p0) - self.offset

        f1 = dehomogenize_3d(robot.mount_T_base_left @ FK_tree["J5"] \
                             @ np.vstack((np.array([0.0, 0.0, 0.0]).reshape(-1,1),1))).reshape(-1)
        p1,_ = cv2.projectPoints(f1, rvec, tvec, self.P, self.D)
        p1 = np.squeeze(p1) - self.offset
        image = cv2.line(image, tuple(p0.astype(int)), tuple(p1.astype(int)), 
                                 color, thickness)

        f2 = dehomogenize_3d(robot.mount_T_base_left @ FK_tree["J6"] \
                             @ np.vstack((np.array([0.0, 0.0, 0.0]).reshape(-1,1),1))).reshape(-1)
        p2,_ = cv2.projectPoints(f2, rvec, tvec, self.P,self.D)
        p2 = np.squeeze(p2) - self.offset
        image = cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), 
                                 color, thickness)



        f3 = dehomogenize_3d(robot.mount_T_base_left @ FK_tree["J7"] \
                             @ np.vstack((np.array([0.0, 0.0, 0.0]).reshape(-1,1),1))).reshape(-1)
        p3,_ = cv2.projectPoints(f3, rvec, tvec, self.P, self.D)
        p3 = np.squeeze(p3) - self.offset
        image = cv2.line(image, tuple(p2.astype(int)), tuple(p3.astype(int)), 
                                 color, thickness)

        image = cv2.circle(image,tuple(p2.astype(int)), 10, (255,0,0), -1)

        return image

    def overlay_points(self, image, points_predicted, color = (255,255,0), thickness = 10):

        points_predicted = points_predicted.astype(int)

        # Printing as a circle
        for i in range(len(points_predicted)):

            points = points_predicted[i]
            image = cv2.circle(image,tuple(points), thickness, color, -1)

        return image
