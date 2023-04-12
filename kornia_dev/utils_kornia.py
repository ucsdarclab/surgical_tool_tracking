import numpy as np
import cv2
import imutils
import math
from scipy.cluster.hierarchy import fclusterdata
import torch
import kornia as K
from scipy.spatial import distance_matrix
from sklearn import linear_model
import scipy.stats
import kornia.feature as KF
#from PIL import Image
import pandas as pd
import copy

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
def drawPolarLines(img = None, lines = None, color = (0, 0, 255)):
    lines = lines.reshape(-1, 2)
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
        cv2.line(img, pt1, pt2, color, 1)
    
    return img

def detectCannyShaftLines(img = None, 
                          hough_rho_accumulator = None, 
                          hough_theta_accumulator = None, 
                          hough_vote_threshold = None,
                          rho_cluster_distance = None,
                          theta_cluster_distance = None,
                          draw_lines = False
                          ):

    # pre-processing
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, ksize=(25,25), sigmaX=0)
    thresh, mask = cv2.threshold(blur, thresh = 150, maxval = 175, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, threshold1 = 200, threshold2 = 255, apertureSize = 5, L2gradient = True)
    edges_and_mask = cv2.bitwise_and(edges, mask)

   # detect lines
    lines = cv2.HoughLinesWithAccumulator(edges_and_mask, rho = hough_rho_accumulator, theta = hough_theta_accumulator, threshold = hough_vote_threshold) 
    if (lines is None):
        return [], img
    print(lines.shape)
    lines = np.squeeze(lines)
    # sort by max votes
    print('in canny shaft lines')
    print('lines: {}'.format(lines))
    print('lines.shape: {}'.format(lines.shape))
    cv2.imwrite('error_img.jpg', img)
    lines = np.reshape(lines, (-1, 3))
    print('lines: {}'.format(lines))
    print('lines.shape: {}'.format(lines.shape))
    sorted_lines = lines[(-lines[:, 2]).argsort()]
    print('sorted_lines: {}'.format(sorted_lines))
    print('sorted_lines.shape: {}'.format(sorted_lines.shape))

    # sort by max votes
    sorted_lines = lines[(-lines[:, 2]).argsort()]
    print('sorted_lines: {}'.format(sorted_lines))
    print('sorted_lines.shape: {}'.format(sorted_lines.shape))
    rhos = sorted_lines[:, 0].reshape(-1, 1)
    print('rhos: {}'.format(rhos))
    print('rhos.shape: {}'.format(rhos.shape))
    thetas = sorted_lines[:, 1].reshape(-1, 1)
    print('thetas: {}'.format(thetas))
    print('thetas: {}'.format(thetas.shape))

    best_lines = []
    if ((rhos.shape == (1, 1)) and (thetas.shape == (1, 1))):
        best_lines.append([float(rhos), float(thetas)])
    else:
        rho_clusters = fclusterdata(rhos, t = rho_cluster_distance, criterion = 'distance', method = 'complete')
        print('rho_clusters: {}'.format(rho_clusters))
        print('rhos_clusters.shape: {}'.format(rho_clusters.shape))
        theta_clusters = fclusterdata(thetas, t = theta_cluster_distance, criterion = 'distance', method = 'complete')
        print('theta_clusters: {}'.format(theta_clusters))
        print('theta_clusters.shape: {}'.format(theta_clusters.shape))

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
    if (draw_lines):
        img = drawPolarLines(img, best_lines[:, 0:2], color = (0, 0, 255))

    # returns Nx2 array of # N detected lines x [rho, theta], img with lines drawn, edges and mask
    return best_lines[:, 0:2], img

# canny image augmentation for kornia network
def cannyPreProcess(img = None):

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, ksize=(25,25), sigmaX=0)
    thresh, mask = cv2.threshold(blur, thresh = 150, maxval = 175, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, threshold1 = 200, threshold2 = 255, apertureSize = 5, L2gradient = True)
    edges_and_mask = cv2.bitwise_and(edges, mask)
    edges_and_mask = cv2.cvtColor(edges_and_mask, cv2.COLOR_GRAY2RGB)
    return edges_and_mask

def centerCrop(img = None, crop_scale = None):
    height, width = img.shape[0], img.shape[1]
    mid_y, mid_x = int(height / 2), int(width / 2)

    crop_height, crop_width = int((height * crop_scale) / 2), int((width * crop_scale) / 2)
    y_offset = mid_y - crop_height
    x_offset = mid_x - crop_width
    crop_img = img[y_offset: (mid_y + crop_height), x_offset: (mid_x + crop_width)]
    return(crop_img, y_offset, x_offset)

# accepts single img and torch[2, 2, 2] tensor of line segment endpoints
# returns altered img
def drawLineSegments(img = None, lines = None, colors = [(0, 0, 0), (255, 255, 255)]):
    
    # BGR (255, 0, 0) = Blue
    for i in range(lines.shape[0]):
        endpoints = lines[i, :, :]
        y1 = int(endpoints[0, 0])
        x1 = int(endpoints[0, 1])
        y2 = int(endpoints[1, 0])
        x2 = int(endpoints[1, 1])
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.line(img, pt1, pt2, colors[i], 5)
    
    return img

# annotate image with pixels
def drawPoints(img = None, point_clouds = None):

    for cloud in point_clouds:
        
        # data
        X = cloud[:, 1].reshape(-1, 1)
        Y = cloud[:, 0].reshape(-1, 1)

        for i in range(X.shape[0]):
            x = int(X[i])
            y = int(Y[i])

            img = cv2.circle(img, center = (x, y), radius = 2, color = (100, 0, 100), thickness = -1)
    
    return img

def fitRansacLines(point_clouds, ransac_params):
    
    # ransac params
    min_samples = int(ransac_params['min_samples'])
    residual_threshold = ransac_params['residual_threshold']
    max_trials = ransac_params['max_trials']
    img_dims = ransac_params['img_dims']
    
    # fit point clouds
    lines = []
    for cloud in point_clouds:
        
        # data
        X = cloud[:, 1].reshape(-1, 1)
        y = cloud[:, 0].reshape(-1, 1)
        residual_threshold *= scipy.stats.median_abs_deviation(y, axis = None)
        ransac = linear_model.RANSACRegressor(min_samples = min_samples, residual_threshold = residual_threshold, max_trials = max_trials)
        
        # sequential ransac
        while (True):
            try: 
                ransac.fit(X, y)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                m = float(ransac.estimator_.coef_)
                b = float(ransac.estimator_.intercept_)
                y1, x1 = int(b), 0
                y2, x2 = int(m * img_dims[1] + b), img_dims[1]
                theta = np.arctan2((x1 - x2), (y2 - y1))
                rho = x1 * np.cos(theta) + y1 * np.sin(theta)

                # convert to hough line bounds
                if (theta < 0) and (theta >= -np.pi):
                    theta += np.pi
                    rho *= -1
                elif (theta < -np.pi) and (theta >= -2 * np.pi):
                    theta += 2 * np.pi
                    rho = rho
                elif (theta >= np.pi) and (theta <= 2 * np.pi):
                    theta -= np.pi
                    rho *= -1
                
                lines.append([rho, theta])
                X = X[outlier_mask]
                y = y[outlier_mask]
                if (len(X) <= min_samples or len(y) <= min_samples):
                    break
            except:
                pass
    return lines

def detectShaftLines(annotated_img = None, 
                    non_annotated_img = None,
                    ref_img = None,
                    ref_tensor = None,
                    crop_ref_lines = None,
                    crop_ref_lines_idx = None,
                    crop_ref_lines_selected = None,
                    model = None, 
                    draw_lines = None,
                    canny_params = {},
                    kornia_params = {}):
    
    # use canny edge detection
    canny_lines = None
    polar_lines_detected_endpoints = None
    intensity_endpoint_clouds = None
    intensity_endpoint_lines = None
    intensity_line_clouds = None
    intensity_line_lines = None
    if ((canny_params is not None) and (canny_params['use_canny'])):
        
        # check that all params are used
        assert(None not in canny_params.values())

        hough_rho_accumulator = canny_params['hough_rho_accumulator']
        hough_theta_accumulator = canny_params['hough_theta_accumulator']
        hough_vote_threshold = canny_params['hough_vote_threshold']
        rho_cluster_distance = canny_params['rho_cluster_distance']
        theta_cluster_distance = canny_params['theta_cluster_distance']
        
        # returns Nx2 array of # N detected lines x [rho, theta]
        # img with lines drawn in color (0, 0, 255)
        canny_lines, annotated_img = detectCannyShaftLines(
                        img = non_annotated_img, 
                        hough_rho_accumulator = hough_rho_accumulator, 
                        hough_theta_accumulator = hough_theta_accumulator, 
                        hough_vote_threshold = hough_vote_threshold,
                        rho_cluster_distance = rho_cluster_distance,
                        theta_cluster_distance = theta_cluster_distance,
                        draw_lines = draw_lines
                        )
    else: # use kornia
        assert((kornia_params is not None) and (kornia_params['use_kornia']))

        # process input image
        img_height = non_annotated_img.shape[0]
        img_width = non_annotated_img.shape[1]
        non_annotated_tensor = K.image_to_tensor(non_annotated_img).float() / 255.0  # [0, 1] [3, crop_dims] float32
        non_annotated_tensor = K.color.rgb_to_grayscale(non_annotated_tensor) # [0, 1] [1, crop_dims] float32
        tensors = torch.stack([ref_tensor, non_annotated_tensor], )
        with torch.inference_mode():
            outputs = model(tensors)
        
        # detect line segments
        line_seg1 = outputs["line_segments"][0]
        line_seg2 = outputs["line_segments"][1]
        desc1 = outputs["dense_desc"][0]
        desc2 = outputs["dense_desc"][1]
        line_heatmap1 = np.asarray(outputs['line_heatmap'][0])
        line_heatmap2 = np.asarray(outputs['line_heatmap'][1])

        # perform association between All line segments 
        # in ref_img and new_img
        with torch.inference_mode():
            matches = model.match(line_seg1, line_seg2, desc1[None], desc2[None])
        valid_matches = matches != -1
        match_indices = matches[valid_matches]

        matched_lines1 = line_seg1[valid_matches]
        #assert(np.allclose(np.asarray(matched_lines1), np.asarray(crop_ref_lines)))
        matched_lines2 = line_seg2[match_indices]

        # sort
        sort_column = 0
        values, indices = matched_lines1[:, :, sort_column].sort()
        sorted_matched_lines1 = matched_lines1[[[x] for x in range(matched_lines1.shape[0])], indices]
        #assert(np.allclose(np.asarray(sorted_matched_lines1), np.asarray(crop_ref_lines_sorted)))
        values, indices = matched_lines2[:, :, sort_column].sort()
        sorted_matched_lines2 = matched_lines2[[[x] for x in range(matched_lines2.shape[0])], indices]

        # find matches to target reference lines
        #print('crop_ref_lines_selected: {}'.format(crop_ref_lines_selected))
        dist_matrix = torch.cdist(torch.flatten(torch.as_tensor(crop_ref_lines_selected), start_dim = 1), torch.flatten(sorted_matched_lines1, start_dim = 1))
        ind = torch.argmin(dist_matrix, dim = 1)
        selected_lines1 = sorted_matched_lines1[ind]
        #print('selected_lines1: {}'.format(selected_lines1))
        #assert(np.allclose(np.asarray(selected_lines1), np.asarray(crop_ref_lines_selected), atol = 1.0, rtol = 0))
        #assert(np.allclose(np.asarray(ind), np.asarray(crop_ref_lines_idx)))
        selected_lines2 = sorted_matched_lines2[ind]

        # select only matching line segments that correspond to ref lines
        if (draw_lines):
            ref_img = drawLineSegments(ref_img, selected_lines1)
            annotated_img = drawLineSegments(annotated_img, selected_lines2)

        # new image detected endpoints
        detected_endpoints = np.asarray(np.around(np.asarray(selected_lines2), decimals = 0), dtype = int) # [[y, x], [y, x]]

        # convert detected endpoints to rho, theta form
        endpoints_to_polar = kornia_params['endpoints_to_polar'] # boolean
        polar_lines_detected_endpoints = None
        if (endpoints_to_polar):
            polar_lines_detected_endpoints = []
            for line in detected_endpoints:
                y1 = line[0][0]
                x1 = line[0][1]
                y2 = line[1][0]
                x2 = line[1][1]

                theta = np.arctan2((x1 - x2), (y2 - y1))
                rho = x1 * np.cos(theta) + y1 * np.sin(theta)
                polar_lines_detected_endpoints.append([rho, theta])
                
                annotated_img = drawPoints(annotated_img, detected_endpoints)
                
                if (draw_lines):
                    annotated_img = drawPolarLines(annotated_img, np.asarray([rho, theta]))
                

        # search region around detected endpoints for all pixels
        # that meet intensity threshold
        # return point cloud vs. rho, theta best fit ransac lines
        use_endpoint_intensities_only = kornia_params['use_endpoint_intensities_only'] # boolean
        endpoint_intensities_to_polar = kornia_params['endpoint_intensities_to_polar'] # boolean
        search_radius = int(kornia_params['search_radius']) # kernel size for dilation
        intensity_params = kornia_params['intensity_params'] # {'metric': value} {'mean': 0, 'std': 1, 'pct': 10}
        ransac_params = kornia_params['ransac_params'] # ransac params 
        # {'min_samples: 3, 'residual_threshold': None, 'max_trials': 100, 'img_dims': (height, width)}
        
        intensity_endpoint_clouds = None
        intensity_endpoint_lines = None
        if (use_endpoint_intensities_only) or (endpoint_intensities_to_polar): # returns all intensity pixels
            
            intensity_endpoint_clouds = []
            kernel = np.ones((search_radius, search_radius), np.uint8)

            for line in detected_endpoints:
                y1 = line[0][0]
                x1 = line[0][1]
                y2 = line[1][0]
                x2 = line[1][1]

                # convert detected endpoints to endpoint intensity clouds
                blank = np.zeros((img_height, img_width))
                dotted = blank.copy()
                dotted[y1, x1] = 255.0
                dotted[y2, x2] = 255.0

                dotted_dilation = cv2.dilate(dotted, kernel, iterations = 1)
                ys, xs = np.where(dotted_dilation)
                dilated_points = list(zip(list(ys), list(xs)))
                dilated_points_intensities = np.asarray([line_heatmap2[coord[0], coord[1]] for coord in dilated_points])

                metric = intensity_params['use_metric']
                if (metric == 'mean'):
                    intensity_threshold = dilated_points_intensities.mean()
                elif (metric == 'std'):
                    stds = intensity_params[metric]
                    intensity_threshold = dilated_points_intensities.mean() + (stds * dilated_points_intensities.std())
                elif (metric == 'pct'):
                    pct = float(intensity_params[metric])
                    intensity_threshold = np.percentile(dilated_points_intensities, pct)
                
                intensity_mask = dilated_points_intensities >= intensity_threshold

                thresholded_dilated_points = np.asarray(dilated_points)[intensity_mask]
                intensity_endpoint_clouds.append(thresholded_dilated_points)

                # draw point clouds
                annotated_img = drawPoints(annotated_img, intensity_endpoint_clouds)

                if (endpoint_intensities_to_polar):
                    intensity_endpoint_lines = fitRansacLines(intensity_endpoint_clouds, ransac_params)
                    if (draw_lines):
                        annotated_img = drawPolarLines(annotated_img, np.asarray(intensity_endpoint_lines))
        
        # search region between detected endpoints for all pixels
        # that meet intensity threshold
        # return point cloud vs. rho, theta best fit ransac lines
        use_line_intensities_only = kornia_params['use_line_intensities_only'] # boolean
        line_intensities_to_polar = kornia_params['line_intensities_to_polar'] # boolean
        intensity_line_clouds = None
        intensity_line_lines = None

        if (use_line_intensities_only) or (line_intensities_to_polar): # returns all intensity pixels
            
            intensity_line_clouds = []
            kernel = np.ones((search_radius, search_radius), np.uint8)

            for line in detected_endpoints:
                y1 = line[0][0]
                x1 = line[0][1]
                y2 = line[1][0]
                x2 = line[1][1]

                # convert detected endpoints to line intensity cloud
                blank = np.zeros((img_height, img_width))
                lined = blank.copy()
                lined = cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), thickness = 1)
                lined_dilation = cv2.dilate(lined, kernel, iterations=1)
                ys, xs = np.where(lined_dilation)
                dilated_line = list(zip(list(ys), list(xs)))
                dilated_line_intensities = np.asarray([line_heatmap2[coord[0], coord[1]] for coord in dilated_line])

                metric = intensity_params['use_metric']
                if (metric == 'mean'):
                    intensity_threshold = dilated_line_intensities.mean()
                elif (metric == 'std'):
                    stds = intensity_params[metric]
                    intensity_threshold = dilated_line_intensities.mean() + (stds * dilated_line_intensities.std())
                elif (metric == 'pct'):
                    pct = float(intensity_params[metric])
                    intensity_threshold = np.percentile(dilated_line_intensities, pct)

                intensity_mask = dilated_line_intensities >= intensity_threshold

                thresholded_dilated_line = np.asarray(dilated_line)[intensity_mask]
                intensity_line_clouds.append(thresholded_dilated_line)

                # draw point clouds
                annotated_img = drawPoints(annotated_img, intensity_endpoint_clouds)

                if (line_intensities_to_polar):
                    intensity_line_lines = fitRansacLines(intensity_line_clouds, ransac_params)
                    if (draw_lines):
                        annotated_img = drawPolarLines(annotated_img, np.asarray(intensity_line_lines))
    
    output = {
        'ref_img': ref_img,
        'new_img': annotated_img,
        'canny_lines': canny_lines,
        'polar_lines_detected_endpoints': polar_lines_detected_endpoints,
        'intensity_endpoint_clouds': intensity_endpoint_clouds,
        'intensity_endpoint_lines': intensity_endpoint_lines,
        'intensity_line_clouds': intensity_line_clouds,
        'intensity_line_lines': intensity_line_lines
    }

    return output

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
    projected_lines = cam.projectShaftLines(p_c, d_c, r)

    # (B, G, R)
    img_l = drawPolarLines(img_list[0], projected_lines[0], (0, 255, 0))
    img_r = drawPolarLines(img_list[1], projected_lines[1], (0, 255, 0))

    return img_l, img_r

def makeShaftAssociations(
                        new_img = None, 
                        ref_tensor = None,
                        ref_img = None,
                        crop_ref_lines = None,
                        crop_ref_lines_sorted = None,
                        crop_ref_lines_selected = None,
                        crop_ref_lines_idx = None,
                        model = None
                        ):
    
    # process input image
    new_tensor = K.image_to_tensor(new_img).float() / 255.0  # [0, 1] [3, crop_dims] float32
    new_tensor = K.color.rgb_to_grayscale(new_tensor) # [0, 1] [1, crop_dims] float32
    tensors = torch.stack([ref_tensor, new_tensor], )
    with torch.inference_mode():
        outputs = model(tensors)
    
    # detect line segments
    line_seg1 = outputs["line_segments"][0]
    line_seg2 = outputs["line_segments"][1]
    desc1 = outputs["dense_desc"][0]
    desc2 = outputs["dense_desc"][1]
    line_heatmap1 = np.asarray(outputs['line_heatmap'][0])
    line_heatmap2 = np.asarray(outputs['line_heatmap'][1])

    # perform association between All line segments 
    # in ref_img and new_img
    with torch.inference_mode():
        matches = model.match(line_seg1, line_seg2, desc1[None], desc2[None])
    valid_matches = matches != -1
    match_indices = matches[valid_matches]

    matched_lines1 = line_seg1[valid_matches]
    #assert(np.allclose(np.asarray(matched_lines1), np.asarray(crop_ref_lines)))
    matched_lines2 = line_seg2[match_indices]

    # sort
    sort_column = 0
    values, indices = matched_lines1[:, :, sort_column].sort()
    sorted_matched_lines1 = matched_lines1[[[x] for x in range(matched_lines1.shape[0])], indices]
    #assert(np.allclose(np.asarray(sorted_matched_lines1), np.asarray(crop_ref_lines_sorted)))
    values, indices = matched_lines2[:, :, sort_column].sort()
    sorted_matched_lines2 = matched_lines2[[[x] for x in range(matched_lines2.shape[0])], indices]

    # find matches to target reference lines
    #print('crop_ref_lines_selected: {}'.format(crop_ref_lines_selected))
    dist_matrix = torch.cdist(torch.flatten(torch.as_tensor(crop_ref_lines_selected), start_dim = 1), torch.flatten(sorted_matched_lines1, start_dim = 1))
    ind = torch.argmin(dist_matrix, dim = 1)
    selected_lines1 = sorted_matched_lines1[ind]
    #print('selected_lines1: {}'.format(selected_lines1))
    #assert(np.allclose(np.asarray(selected_lines1), np.asarray(crop_ref_lines_selected), atol = 1.0, rtol = 0))
    #assert(np.allclose(np.asarray(ind), np.asarray(crop_ref_lines_idx)))
    selected_lines2 = sorted_matched_lines2[ind]
    
    # pick only shaft lines in reference image by length (largest 2x lines)
    #reference_line_lengths = []
    #for i in range(matched_lines1.shape[0]):
        #reference_line_lengths.append(np.linalg.norm(matched_lines1[i][0] - matched_lines1[i][1]))
    #reference_line_lengths = np.asarray(reference_line_lengths)
    #ind = np.argpartition(reference_line_lengths, -2)[-2:]
    
    ref_img = drawLineSegments(ref_img, selected_lines1)
    new_img = drawLineSegments(new_img, selected_lines2)

    return ref_img, new_img
