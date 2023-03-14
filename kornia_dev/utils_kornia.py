import numpy as np
import cv2
import imutils
import math
from scipy.cluster.hierarchy import fclusterdata
import torch
import kornia as K

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

# accepts single img and Nx2 [rho, theta] array of line parameters
# returns altered img
def drawPolarLines(img, lines, color = (0, 0, 255)):
    for i in range(lines.shape[0]):
        rho = lines[i, 0]
        theta = lines[i, 1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        # BGR (255, 0, 0) = Blue
        cv2.line(img, pt1, pt2, color, 2)
    
    return img

def detectCannyShaftLines(img):

    # pre-processing
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, ksize=(25,25), sigmaX=0)
    thresh, mask = cv2.threshold(blur, thresh = 150, maxval = 175, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, threshold1 = 200, threshold2 = 255, apertureSize = 5, L2gradient = True)
    edges_and_mask = cv2.bitwise_and(edges, mask)

   # detect lines
    lines = cv2.HoughLinesWithAccumulator(edges_and_mask, rho = 5, theta = 0.09, threshold = 100) 
    lines = np.squeeze(lines)
    # sort by max votes
    sorted_lines = lines[(-lines[:, 2]).argsort()]

    # sort by max votes
    sorted_lines = lines[(-lines[:, 2]).argsort()]

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

    best_lines = np.asarray(best_lines)

    # check for negative rho, add 2*pi to theta
    best_lines[:, 1][best_lines[:, 0] < 1] = best_lines[:, 1][best_lines[:, 0] < 1] + 2 * np.pi
    # replace negative rho with abs(rho)
    best_lines[:, 0][best_lines[:, 0] < 0] = best_lines[:, 0][best_lines[:, 0] < 1] * -1

    # eliminate vertical lines
    vertical_line_mask = []
    for i in range(best_lines.shape[0]):
        theta = best_lines[i, 1]
        if (theta > -10 * np.pi / 180) and (theta < 10 * np.pi / 180):
            vertical_line_mask.append(False)
        elif (theta > 170 * np.pi / 180) and (theta < 190 * np.pi / 180):
            vertical_line_mask.append(False)
        else:
            vertical_line_mask.append(True)
    best_lines = best_lines[vertical_line_mask, :]

    # draw all detected and clustered edges
    # (B, G, R)
    img = drawLines(img, best_lines[:, 0:2], color = (0, 0, 255))

    # returns Nx2 array of # N detected lines x [rho, theta], img with lines drawn, edges and mask
    return best_lines[:, 0:2], img

# canny image augmentation for kornia network
def cannyPreProcess_kornia(img):

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, ksize=(25,25), sigmaX=0)
    thresh, mask = cv2.threshold(blur, thresh = 150, maxval = 175, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, threshold1 = 200, threshold2 = 255, apertureSize = 5, L2gradient = True)
    edges_and_mask = cv2.bitwise_and(edges, mask)
    edges_and_mask = cv2.cvtColor(edges_and_mask, cv2.COLOR_GRAY2RGB)
    return edges_and_mask

def center_crop(img, dim):
    height, width = img.shape[0], img.shape[1]
    mid_y, mid_x = int(height / 2), int(width / 2)

    crop_height, crop_width = int(dim[0] / 2), int(dim[1] / 2)
    y_offset = mid_y - crop_height
    x_offset = mid_x - crop_width
    crop_img = img[y_offset: (mid_y + crop_height), x_offset: (mid_x + crop_width)]
    return(crop_img, y_offset, x_offset)

# accepts single img and torch[2, 2, 2] tensor of line segment endpoints
# returns altered img
def drawLineSegments(img, lines):
    # BGR (255, 0, 0) = Blue
    colors = [(255, 0, 0), (0, 0, 255)]
    for i in range(lines.shape[0]):
        endpoints = lines[i, :, :]
        y1 = int(endpoints[0, 0])
        x1 = int(endpoints[0, 1])
        y2 = int(endpoints[1, 0])
        x2 = int(endpoints[1, 1])
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.line(img, pt1, pt2, colors[i], 2)
    
    return img

def detectShaftLines_kornia(new_img, ref_img, crop_ref_lines, crop_ref_dims, model, use_intensity, intensity_radius = 3):
    new_img, y_offset, x_offset = center_crop(new_img, crop_ref_dims) # crop_dims x 3 (RGB) uint8 ndarray [0 255]
    new_img = K.image_to_tensor(new_img).float() / 255.0  # [0, 1] [3, crop_dims] float32
    new_img = K.color.rgb_to_grayscale(new_img) # [0, 1] [1, crop_dims] float32
    imgs = torch.stack([ref_img, new_img], )
    with torch.inference_mode():
        outputs = model(imgs)
    
    # detect line segments
    line_seg1 = outputs["line_segments"][0]
    line_seg2 = outputs["line_segments"][1]
    desc1 = outputs["dense_desc"][0]
    desc2 = outputs["dense_desc"][1]

    # perform association
    with torch.inference_mode():
        matches = model.match(line_seg1, line_seg2, desc1[None], desc2[None])
    valid_matches = matches != -1
    match_indices = matches[valid_matches]

    matched_lines1 = line_seg1[valid_matches]
    matched_lines2 = line_seg2[match_indices]

    # sort matched lines by y-coordinate
    sort_column = 0
    values, indices = matched_lines1[:, :, sort_column].sort()
    sorted_matched_lines1 = matched_lines1[[[x] for x in range(matched_lines1.shape[0])], indices]

    # load ref lines and find identical lines in ref_img lines (matched_lines1)
    dist_matrix = torch.cdist(torch.flatten(crop_ref_lines, start_dim = 1), torch.flatten(sorted_matched_lines1, start_dim = 1))
    ind = torch.argmin(dist_matrix, dim = 1)

    # select matching shaft lines
    selected_lines1 = matched_lines1[ind]
    selected_lines2 = matched_lines2[ind] #torch[2, 2, 2]

    # find intensity-based endpoints
    if (use_intensity):
        line_heatmap = np.asarray(outputs['line_heatmap'][1])
        
        if (intensity_radius <= 0):
            intensity_radius = 3

        intensity_endpoints = []
        x_min = 0
        x_max = crop_ref_dims[1]
        y_min = 0
        y_max = crop_ref_dims[0]
        for i in range(selected_lines2.shape[0]):
            endpoints = selected_lines2[i, :, :]
            for j in range(endpoints.shape[0]):
                y = int(endpoints[j][0])
                x = int(endpoints[j][1])
                top_left = (max(y_min, y - intensity_radius), max(x_min, x - intensity_radius)) 
                bottom_right = (min(y_max, y + intensity_radius), min(x_max, x + intensity_radius))
                rows = np.arange(top_left[0], bottom_right[0])
                cols = np.arange(top_left[1], bottom_right[1])
                intensities = line_heatmap[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
                idx_flat = intensities.ravel().argmax()
                idx = np.unravel_index(idx_flat, intensities.shape)
                y = rows[idx[0]]
                x = cols[idx[1]]
                intensity_endpoints.append([y, x])

        selected_lines2 = torch.as_tensor(np.asarray(intensity_endpoints).reshape(selected_lines2.shape))

    # draw matched lines on cropped reference input image
    ref_img = drawLineSegments(ref_img, selected_lines1)

    # draw matched lines on new input img (at original size)
    new_img = drawLineSegments(new_img, selected_lines2)

    # get rho, theta params
    lines = []
    for i in range(selected_lines2.shape[0]):
        endpoints = selected_lines2[i, :, :]
        y1 = int(endpoints[0][0])
        x1 = int(endpoints[0][1])
        y2 = int(endpoints[1][0])
        x2 = int(endpoints[1][1])
        theta = np.arctan2((x1 - x2), (y2 - y1))
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        lines.append([rho, theta])
    lines = np.asarray(lines)
    
    # returns torch[2, 2, 2] tensor of endpoints for 2 line segments
    # returns Nx2 array line segments [[rho, theta], [rho, theta]]
    # returns input image with line segments drawn
    # returns reference image with line segments drawn
    return ([selected_lines2, lines], new_img, ref_img)

def drawShaftLines(shaftFeatures, cam, cam_T_b, img_list):

    # Project points from base to 2D L/R camera image planes
    # Get shaft feature points and lines from FK transform in base frame
    p_b, d_b, _, r = shaftFeatures
    # Transform shaft featurepoints from base frame to camera-to-base frame
    p_c = np.dot(cam_T_b, np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    
    # Rotate directions from base to camera frame (no translation)
    d_c = np.dot(cam_T_b[0:3, 0:3], np.transpose(d_b))
    d_c = np.transpose(d_c)
    
    # Project shaft lines from L and R camera-to-base frames onto 2D camera image plane
    projected_lines = cam.projectPolarShaftLines(p_c, d_c, r)

    # (B, G, R)
    img_l = drawPolarLines(img_list[0], projected_lines[0], (0, 255, 0))
    img_r = drawPolarLines(img_list[1], projected_lines[1], (0, 255, 0))

    return img_l, img_r
