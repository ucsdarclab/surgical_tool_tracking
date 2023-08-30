import numpy as np
import math
from scipy import optimize
from scipy.stats import norm
from scipy.spatial import distance_matrix
from utils_kornia import *


# Example initialization function for ParticleFilter class, kwargs would include std
def sampleNormalDistribution(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    prob_individual = norm.pdf(sample, m, std)
    prob_individual[np.isnan(prob_individual)] = 1.0
    return sample, np.prod(prob_individual)

# probability density at a point in normal distribution
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

# Example motion model function for Particle Filter class, kwargs would include std
def additiveGaussianNoise(state, std):
    sample, prob = sampleNormalDistribution(np.asarray(std))
    return state + sample, prob

# Create "Lumped Error" Motion model that:
#   1. Includes uncertainty from hand-eye transform
#   2. Distributes uncertainty from joint angles
#   3. sate is [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, e_nb, ..., e_n]
#       where pos, ori is position and axis/angle rep of lumped error
#       e_nb+1, ..., e_n are the errors of the tracked joint angles
#       For more details check: https://arxiv.org/pdf/2102.06235.pdf
#   4. robot_arm is RobotLink that is being tracked
def lumpedErrorMotionModel(state, std_pos, std_ori, robot_arm, std_j, nb):
    # Gaussian uncertainty in hand-eye transform and joint angles
    std = np.concatenate((np.array([std_pos, std_pos, std_pos, std_ori, std_ori, std_ori]), std_j))
    sample, prob = sampleNormalDistribution(std)
    T = poseToMatrix(sample[:6] + state[:6])
    
    # propogate joint error from lumped error which is only up to nb jonts
    T = np.dot(T, robot_arm.propogateJointErrorToLumpedError(sample[6:nb+6]))

    # Return the new state (with added uncertainty on the tracked joint angles)
    return np.concatenate((matrixToPose(T), sample[nb+6:] + state[6:])), prob
    
# State is: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, e_nb, ..., e_n]
# where pos, ori is position and axis/angle rep of lumped error
# e_nb+1, ..., e_n are the errors of the tracked joint angles
def pointFeatureObs(state, point_detections, robot_arm, joint_angle_readings, cam, cam_T_b, gamma, association_threshold=20):
    # Get lumped error
    T = poseToMatrix(state[:6])

    # Add estimated joint errors to the robot link
    if state.shape[0] > 6:
        joint_angle_readings[-(state.shape[0]-6):] += state[6:]
    robot_arm.updateJointAngles(joint_angle_readings)

    # Project points
    p_b, point_featuresNames = robot_arm.getPointFeatures()
    #print('robot_arm.getPointFeatures()[1]: {}'.format(point_featuresNames))
    #print('robot_arm.getPointFeatures()[0]: {}'.format(p_b))
    holdout_point_name = 'yaw_1'
    holdout_point_index = point_featuresNames.index(holdout_point_name)

    p_c = np.dot(np.dot(cam_T_b, T), np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    projected_points = cam.projectPoints(p_c)

    #print('len(projected_points): {}'.format(len(projected_points)))
    #print('len(point_detections): {}'.format(len(point_detections)))
    
    # Raise error if number of cameras doesn't line up
    if len(projected_points) != len(point_detections):
        raise ValueError("Length of projected_points is {} but length of points_detections is {}.\n".format(len(projected_points), len(point_detections)) + "Note that these lengths represent the number of cameras being used.")
    
    # Make association between detected and projected & compute probability
    prob = 1
    # len(projected_points) = # of cameras
    # each list in projected points (2x for R/L cameras) is also a list of projected points
    for c_idx, proj_point in enumerate(projected_points):
        #print('c_idx: {}'.format(c_idx))
        #print('proj_point: {}'.format(proj_point))
        #print('proj_point.shape: {}'. format(proj_point.shape))
        #if (c_idx == 1):
            #print('(x, y) tool tip projected point in R camera: {}'.format(proj_point[holdout_point_index, :]))
        proj_point = np.delete(proj_point, obj = holdout_point_index, axis = 0).copy()
        #print('proj_point: {}'.format(proj_point))

        #print('point_detections[c_idx]: {}'.format(point_detections[c_idx]))
        #print('point_detections[c_idx].shape: {}'.format(point_detections[c_idx].shape))
        
        # Use hungarian algorithm to match projected and detected points
        C = np.linalg.norm(proj_point[:, None, :] - point_detections[c_idx][None, :,  :], axis=2)
        row_idx, col_idx = optimize.linear_sum_assignment(C)
        #print('C: {}'.format(C))
        #print('C.shape: {}'.format(C.shape))
        #print('row_idx: {}'.format(row_idx))
        #print('col_idx: {}'.format(col_idx))
        
        # Use threshold to remove outliers
        idx_to_keep = C[row_idx, col_idx] < association_threshold
        row_idx = row_idx[idx_to_keep]
        col_idx = col_idx[idx_to_keep]
        
        #print('C: {}'.format(C))
        #print('C.shape: {}'.format(C.shape))
        #print('row_idx: {}'.format(row_idx))
        #print('row_idx.shape: {}'.format(row_idx.shape))
        #print('col_idx: {}'.format(col_idx))
        #print('col_idx.shape: {}'.format(col_idx.shape))
        #try: 
            #print('holdout point index: {}'.format(list(row_idx).index(holdout_point_index)))
            #detected_holdout_point_index = list(col_idx)[holdout_point_index]
            #print('detected_holdout_point_index: {}'.format(detected_holdout_point_index))
            #detected_holdout_point = point_detections[c_idx][detected_holdout_point_index, :]
            #print('detected_holdout_point: {}'.format(detected_holdout_point))
            #print('projected holdout point: {}'.format(proj_point[holdout_point_index, :]))
        #except ValueError as e:
            #print(e)
            #continue

        #print('C[row_idx, col_idx]: {}'.format(C[row_idx, col_idx]))
        #print('C[row_idx, col_idx].shape: {}'.format(C[row_idx, col_idx].shape))

        # Compute observation probability
        prob *= np.sum(np.exp(-gamma*C[row_idx, col_idx])) \
                + (proj_point.shape[0] - len(row_idx))*np.exp(-gamma*association_threshold)
        
    return prob

# State is: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, e_nb, ..., e_n]
# where pos, ori is position and axis/angle rep of lumped error
# e_nb+1, ..., e_n are the errors of the tracked joint angles
def pointFeatureObsRightLumpedError(state, point_detections, robot_arm, cam, 
                                    cam_T_b, joint_angle_readings, gamma, association_threshold=20):
    # Get lumped error
    T = poseToMatrix(state[:6])

    # Add estimated joint errors to the robot link
    if state.shape[0] > 6:
        joint_angle_readings[-(state.shape[0]-6):] += state[6:]
    robot_arm.updateJointAngles(joint_angle_readings)

    # Project points
    p_b,_ = robot_arm.getPointFeatures()
    p_c = np.dot(np.dot(cam_T_b, T), np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    projected_points = cam.projectPoints(p_c)
    
    # Raise error if number of cameras doesn't line up
    if len(projected_points) != len(point_detections):
        raise ValueError("Length of projected_points is {} but length of points_detections is {}.\n".format(len(projected_points), 
                                                                                                            len(point_detections)) \
                        + "Note that these lengths represent the number of cameras being used.")
    
    # Make association between detected and projected & compute probability
    prob = 1
    for c_idx, proj_point in enumerate(projected_points):
        # Use hungarian algorithm to match projected and detected points
        C = np.linalg.norm(proj_point[:, None, :] - point_detections[c_idx][None, :,  :], axis=2)
        row_idx, col_idx = optimize.linear_sum_assignment(C)
        
        # Use threshold to remove outliers
        idx_to_keep = C[row_idx, col_idx] < association_threshold
        row_idx = row_idx[idx_to_keep]
        col_idx = col_idx[idx_to_keep]
        
        # Compute observation probability
        prob *= np.sum(np.exp(-gamma*C[row_idx, col_idx])) \
                + (proj_point.shape[0] - len(row_idx))*np.exp(-gamma*association_threshold)
        
    return prob

def shaftFeatureObs(state, detected_lines, robot_arm, cam, cam_T_b, joint_angle_readings, gamma_rho, gamma_theta, rho_thresh, theta_thresh):
    #print('in shaftFeatureObs')
    #print('state: {}'.format(state))
    #print('detected_lines: {}'.format(detected_lines))
    #print('cam: {}'.format(cam))
    
    # Get lumped error
    T = poseToMatrix(state[:6])
    #print('T: {}'.format(T))

    # Add estimated joint errors to the robot link
    if state.shape[0] > 6:
        joint_angle_readings[-(state.shape[0]-6):] += state[6:]
    robot_arm.updateJointAngles(joint_angle_readings)

    # Project points from base to 2D L/R camera image planes
    # Get shaft feature points and lines from FK transform in base frame
    p_b, d_b, _, r = robot_arm.getShaftFeatures()
    #print('p_b: {}'.format(p_b))
    #print('p_b.shape: {}'.format(p_b.shape))
    #print('d_b: {}'.format(d_b))
    #print('d_b.shape: {}'.format(d_b.shape))
    #print('r: {}'.format(r))
    # Transform shaft featurepoints from base frame to camera-to-base frame
    p_c = np.dot(np.dot(cam_T_b, T), np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    #print('p_c: {}'.format(p_c))
    #print('p_c.shape: {}'.format(p_c.shape))
    
    # Rotate directions from base to camera frame (no translation)
    d_c = np.dot(np.dot(cam_T_b, T)[0:3, 0:3], np.transpose(d_b))
    d_c = np.transpose(d_c)
    #print('d_c: {}'.format(d_c))
    #print('d_c.shape: {}'.format(d_c.shape))
    
    
    # Project shaft lines from L and R camera-to-base frames onto 2D camera image plane
    assert(cam is not None)
    projected_lines = cam.projectShaftLines(p_c, d_c, r)



        # Raise error if number of cameras doesn't line up
    if len(projected_lines) != len(detected_lines):
        raise ValueError("Length of projected_lines is {} but length of line_detections is {}.\n".format(len(projected_lines), 
                                                                                                            len(detected_lines)) \
                        + "Note that these lengths represent the number of cameras being used.")
    
    # Make association between detected and projected & compute probability
    prob = 1
    association_threshold = gamma_rho * rho_thresh + gamma_theta * theta_thresh
    # len(projected_points) = # of cameras
    # each list in projected points (2x for R/L cameras) is also a list of projected points
    #print('entering cost association calculation')
    #print('projected_lines: {}'.format(projected_lines))
    #print('detected_lines: {}'.format(detected_lines))
    
    for cam_idx, proj_lines in enumerate(projected_lines):
        # Use hungarian algorithm to match projected and detected points
        C_rho = gamma_rho * distance_matrix(proj_lines[:, 0, None], detected_lines[cam_idx][:, 0, None])
        C_theta = gamma_theta * distance_matrix(proj_lines[:, 1, None], detected_lines[cam_idx][:, 1, None])
        C = C_rho + C_theta
        row_idx, col_idx = optimize.linear_sum_assignment(C)
        
        # Use threshold to remove outliers
        idx_to_keep = C[row_idx, col_idx] < association_threshold
        row_idx = row_idx[idx_to_keep]
        col_idx = col_idx[idx_to_keep]
        
        # Compute observation probability
        prob *= np.sum(np.exp(C[row_idx, col_idx])) + (proj_lines.shape[0] - len(row_idx))*np.exp(-1 * association_threshold)
        
    return prob

def shaftFeatureObs_kornia(
        state, 
        use_lines = None, 
        use_clouds = None, 
        detected_lines = None, 
        intensity_clouds = None, 
        robot_arm = None, 
        cam = None, 
        cam_T_b = None, 
        joint_angle_readings = None, 
        cost_assoc_params = None,
        pixel_probability_params = None
        ):
    
    # determine metric
    if (use_lines):
        algo = use_lines
        detected_lines = detected_lines[algo]
        print('shaftfeatureobs detected_lines: {}'.format(detected_lines))
        #print('shaftfeatureobs detected_lines.shape: {}'.format(detected_lines.shape))
        
    elif (use_clouds):
        algo = use_clouds
        intensity_clouds = intensity_clouds[algo]
        print('shaftfeatureobs intensity_clouds: {}'.format(intensity_clouds))
        #print('shaftfeatureobs intensity_clouds.shape: {}'.format(intensity_clouds.shape))

    '''    
    # no lines detected
    if ((detected_lines is None) or (len(detected_lines) == 0)):
        prob = 1
        return prob

    #print('shaftfeatureobs detected_lines: {}'.format(detected_lines))
    #print('shaftfeatureobs detected_lines[0].shape: {}'.format(detected_lines[0].shape))
    #print('shaftfeatureobs detected_lines[1].shape: {}'.format(detected_lines[1].shape))
    for detected_line in detected_lines:
        if (detected_line is None):
            prob = 1
            return prob
        if (detected_line.size == 0):
            prob = 1
            return prob
        if (np.any(detected_line) is False):
            prob = 1
            return prob
        if (detected_line.shape == ()):
            prob = 1
            return prob
    '''

    # Get lumped error
    T = poseToMatrix(state[:6])
    #print('T: {}'.format(T))

    # Add estimated joint errors to the robot link
    if state.shape[0] > 6:
        joint_angle_readings[-(state.shape[0]-6):] += state[6:]
    robot_arm.updateJointAngles(joint_angle_readings)

    # Project points from base to 2D L/R camera image planes
    # Get shaft feature points and lines from FK transform in base frame
    p_b, d_b, _, r = robot_arm.getShaftFeatures()
    #print('p_b: {}'.format(p_b))
    #print('p_b.shape: {}'.format(p_b.shape))
    #print('d_b: {}'.format(d_b))
    #print('d_b.shape: {}'.format(d_b.shape))
    #print('r: {}'.format(r))
    # Transform shaft featurepoints from base frame to camera-to-base frame
    p_c = np.dot(np.dot(cam_T_b, T), np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    #print('p_c: {}'.format(p_c))
    #print('p_c.shape: {}'.format(p_c.shape))
    
    # Rotate directions from base to camera frame (no translation)
    d_c = np.dot(np.dot(cam_T_b, T)[0:3, 0:3], np.transpose(d_b))
    d_c = np.transpose(d_c)
    #print('d_c: {}'.format(d_c))
    #print('d_c.shape: {}'.format(d_c.shape))
    
    
    # Project shaft lines from L and R camera-to-base frames onto 2D camera image plane
    assert(cam is not None)
    projected_lines = cam.projectShaftLines(p_c, d_c, r)
    #print('shaftfeatureobs projected lines: {}'.format(projected_lines))
    #print('shaftfeatureobs projected lines.shape: {}'.format(projected_lines.shape))

        # Raise error if number of cameras doesn't line up
    '''
    if len(projected_lines) != len(detected_lines):
        raise ValueError("Length of projected_lines is {} but length of line_detections is {}.\n".format(len(projected_lines), 
                                                                                                            len(detected_lines)) \
                        + "Note that these lengths represent the number of cameras being used.")
    '''
    # Make association between detected and projected lines
    # compute probability of detected lines
    if (use_lines):
        prob = 1
        association_threshold = cost_assoc_params['gamma_rho'] * cost_assoc_params['rho_thresh'] +\
                                cost_assoc_params['gamma_theta'] * cost_assoc_params['theta_thresh']
        # len(projected_points) = # of cameras
        # each list in projected points (2x for R/L cameras) is also a list of projected points
        for cam_idx, proj_lines in enumerate(projected_lines):
            #print('shaftfeatureobs hungarian proj_lines: {}'.format(proj_lines))
            #print('shaftfeatureobs hungarian proj_lines.shape: {}'.format(proj_lines.shape))
            #print('shaftfeatureobs hungarian detected_lines[cam_idx]: {}'.format(detected_lines[cam_idx]))
            #print('shaftfeatureobs hungarian detected_lines[cam_idx].shape: {}'.format(detected_lines[cam_idx].shape))
            # Use hungarian algorithm to match projected and detected points
            C_rho = cost_assoc_params['gamma_rho'] * distance_matrix(proj_lines[:, 0, None], detected_lines[cam_idx][:, 0, None])
            C_theta = cost_assoc_params['gamma_theta'] * distance_matrix(proj_lines[:, 1, None], detected_lines[cam_idx][:, 1, None])
            C = C_rho + C_theta
            row_idx, col_idx = optimize.linear_sum_assignment(C)
            
            # Use threshold to remove outliers
            idx_to_keep = C[row_idx, col_idx] < association_threshold
            row_idx = row_idx[idx_to_keep]
            col_idx = col_idx[idx_to_keep]
            
            # Compute observation probability
            prob *= np.sum(np.exp(C[row_idx, col_idx])) + (proj_lines.shape[0] - len(row_idx))*np.exp(-1 * association_threshold)
    
    # compute probability of intensity point clouds
    elif (use_clouds):
        prob = 1
        sigma2_x = pixel_probability_params['sigma2_x']
        sigma2_y = pixel_probability_params['sigma2_y']

        intensity_clouds_l = intensity_clouds[0]
        intensity_clouds_r = intensity_clouds[1]

        projected_lines_l = projected_lines[0]
        projected_lines_r = projected_lines[1]

        distributions_l = []
        for proj_line_l in projected_lines_l:
            rho = proj_line_l[0]
            theta = proj_line_l[1]
            mean = 0
            var = (np.cos(theta) ** 2) * sigma2_x + (np.sin(theta) ** 2) * sigma2_y
            distribution = [rho, theta, mean, var]
            distributions_l.append(distribution)

        distributions_r = []
        for proj_line_r in projected_lines_r:
            rho = proj_line_r[0]
            theta = proj_line_r[1]
            mean = 0
            var = (np.cos(theta) ** 2) * sigma2_x + (np.sin(theta) ** 2) * sigma2_y
            distribution = [rho, theta, mean, var]
            distributions_r.append(distribution)
        
        # flatten detected clouds to list of x, y points
        intensity_clouds_l = np.vstack(intensity_clouds_l)
        assert(intensity_clouds_l.shape[1] == 2)
        
        max_point_probs_l = []
        for i in range(intensity_clouds_l.shape[0]):
            y = intensity_clouds_l[i, 0]
            x = intensity_clouds_l[i, 1]
            max_point_prob = -99
            for distribution in distributions_l:
                rho = distribution[0]
                theta = distribution[1]
                mean = distribution[2]
                var = distribution[3]
                rv = x * np.cos(theta) + y * np.sin(theta) - rho
                point_prob = normpdf(rv, mean, np.sqrt(var))
                if (point_prob > max_point_prob):
                    max_point_prob = point_prob
            max_point_probs_l.append(max_point_prob)
            # color pixel to match projected line
        
        # flatten detected clouds to list of x, y points
        intensity_clouds_r = np.vstack(intensity_clouds_r)
        assert(intensity_clouds_r.shape[1] == 2)
        
        max_point_probs_r = []
        for i in range(intensity_clouds_r.shape[0]):
            y = intensity_clouds_r[i, 0]
            x = intensity_clouds_r[i, 1]
            max_point_prob = -99
            for distribution in distributions_r:
                rho = distribution[0]
                theta = distribution[1]
                mean = distribution[2]
                var = distribution[3]
                rv = x * np.cos(theta) + y * np.sin(theta) - rho
                point_prob = normpdf(rv, mean, np.sqrt(var))
                if (point_prob > max_point_prob):
                    max_point_prob = point_prob
            max_point_probs_r.append(max_point_prob)
        
        # join all pixel probabilities from l/r cameras
        max_point_probs = []
        max_point_probs.extend(max_point_probs_l)
        max_point_probs.extend(max_point_probs_r)
        max_point_probs = np.asarray(max_point_probs)
        prob *= np.prod(max_point_probs)

    return prob