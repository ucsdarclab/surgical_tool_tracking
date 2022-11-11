import numpy as np
import cv2
import imutils
import math
from scipy.cluster.hierarchy import fclusterdata

def projectSkeleton(skeletonPts3D, cam_T_b, img_list, project_point_function):
    # skeletonPts3D should be in the same format as getSkeletonPoints from RobotLink
    # img_list
    for skeletonPairs in skeletonPts3D:
        pt_pair = np.transpose(np.array(skeletonPairs))
        pt_pair = np.concatenate((pt_pair, np.ones((1, pt_pair.shape[1]))))
        pt_pair = np.transpose(np.dot(cam_T_b, pt_pair)[:-1,:])
        proj_pts_list = project_point_function(pt_pair)
        
        if len(img_list) != len(proj_pts_list):
            raise ValueError("Number of imgs inputted must equal the project_point_function cameras")
        
        for idx, proj_pts in enumerate(proj_pts_list):
            try:
                img_list[idx] = cv2.line(img_list[idx], (int(proj_pts[0,0]), int(proj_pts[0,1])), 
                                   (int(proj_pts[1,0]), int(proj_pts[1,1])),  (0,255,0), 5)
            except:
                continue

    return img_list

# TODO: draw edges
# def drawEdges()

def axisAngleToRotationMatrix(axis_angle):
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-5:
        return np.eye(3)
    
    axis  = axis_angle/angle
    cross_product_mat_axis = np.array([[0, -axis[2], axis[1]],
                                       [axis[2], 0, -axis[0]],
                                       [-axis[1], axis[0], 0]])
    return np.cos(angle)*np.eye(3) + np.sin(angle) * cross_product_mat_axis \
            + (1.0 - np.cos(angle))*np.outer(axis,axis)

def rotationMatrixToAxisAngle(rotation_matrix):
    angle = np.arccos((rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2] - 1.0)/2.0 )
    axis  = np.array([rotation_matrix[2,1] - rotation_matrix[1,2], 
                      rotation_matrix[0,2] - rotation_matrix[2,0],
                      rotation_matrix[1,0] - rotation_matrix[0,1]])
    axis = axis/np.linalg.norm(axis)
    return axis*angle

def poseToMatrix(pose):
    # Pose is [position, ori]
    T = np.eye(4)
    T[:-1, -1] = np.array(pose[:3])
    T[:-1, :-1] = axisAngleToRotationMatrix(pose[3:])
    return T

def matrixToPose(T):
    pose = np.zeros((6,))
    pose[:3] = T[:-1, -1]
    pose[3:] = rotationMatrixToAxisAngle(T[:-1, :-1])
    return pose


def invertTransformMatrix(T):
    out = np.eye(4)
    out[:-1, :-1] = np.transpose(T[:-1, :-1])
    out[:-1,  -1] = -np.dot(out[:-1, :-1], T[:-1,  -1])
    return out

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


def detectShaftLines(img, draw_lines=True, show_canny=False, show_canny_name='canny'):

    # pre-processing
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # TODO: switch with bilateral filter
    blur = cv2.GaussianBlur(grey, ksize=(25,25), sigmaX=0)
    thresh, mask = cv2.threshold(blur, thresh = 150, maxval = 175, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, threshold1 = 200, threshold2 = 255, apertureSize = 5, L2gradient = True)
    edges_and_mask = cv2.bitwise_and(edges, mask)

    if (show_canny):
        cv2.imshow(show_canny_name, edges_and_mask)

    # detect lines
    lines = cv2.HoughLinesWithAccumulator(edges_and_mask, rho = 5, theta = 0.09, threshold = 150)
    if (lines is not None): 
        lines = np.squeeze(lines)
    else:
        return None, img

    # sort by max votes
    sorted_lines = lines[(-lines[:, 2]).argsort()]
    # draw all found lines
    if (draw_lines):
        for i in range(sorted_lines.shape[0]):
            rho = sorted_lines[i, 0]
            theta = sorted_lines[i, 1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 2)

    # cluster by euclidean distance
    rho_clusters = fclusterdata(sorted_lines[:, 0].reshape(-1, 1), t = 5, criterion = 'distance', method = 'complete')
    theta_clusters = fclusterdata(sorted_lines[:, 1].reshape(-1, 1), t = 0.09, criterion = 'distance', method = 'complete')

    best_lines = []
    checked_clusters = []
    for i in range(sorted_lines.shape[0]):
        rho_cluster = rho_clusters[i]
        theta_cluster = theta_clusters[i]
        cluster = (rho_cluster, theta_cluster)
        if (cluster in checked_clusters):
            continue
        best_lines.append([lines[i, 0], lines[i, 1]])
        checked_clusters.append(cluster)
        
        # keep all candidate lines
        #if len(best_lines) >= 2:
            #break

    best_lines = np.asarray(best_lines)

    if (draw_lines):
        for i in range(best_lines.shape[0]):
            rho = best_lines[i, 0]
            theta = best_lines[i, 1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(img, pt1, pt2, (255,0,0), 2)

    '''
    for (auto &i : lines) {

        if(i[0] < 0){
            i[0] = -i[0];
            i[1] += 3.14159265359;
        }
        float rho = i[0], theta = i[1];
        //TUNE: we might not want lines that are super horizontal or vertical
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        int x1 = int(x0 + 1000*(-b)), y1 = int(y0 + 1000*(a)), x2 = int(x0 - 1000*(-b)), y2 = int(y0 - 1000*(a));
        cv::line(out_img, cv::Point(x1, y1), cv::Point(x2, y2), m_detected_contour_color, 2);
    }
    '''

    return best_lines[:, 0:2], img
