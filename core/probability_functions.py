import numpy as np
from scipy import optimize
from scipy.stats import norm
from .utils import *


# Example initialization function for ParticleFilter class, kwargs would include std
def sampleNormalDistribution(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    prob_individual = norm.pdf(sample, m, std)
    prob_individual[np.isnan(prob_individual)] = 1.0
    return sample, np.prod(prob_individual)

# Example motion model function for Particle Filter class, kwargs would include std
def additiveGaussianNoise(state, std):
    sample, prob = sampleNormalDistribution(std)
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
    # len(projected_points) = # of cameras
    # each list in projected points (2x for R/L cameras) is also a list of projected points
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



# State is: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, e_nb, ..., e_n]
# where pos, ori is position and axis/angle rep of lumped error
# e_nb+1, ..., e_n are the errors of the tracked joint angles
def pointFeatureObsRightLumpedError(state, point_detections, robot_arm, joint_angle_readings, cam, 
                                    cam_T_b, gamma, association_threshold=20):
    # Get lumped error
    T = poseToMatrix(state[:6])

    # Add estimated joint errors to the robot link
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

def shaftFeatureObs(state, line_detections, robot_arm, joint_angle_readings, cam, cam_T_b, gamma_rho, gamma_theta, rho_thresh, theta_thresh):
    # Get lumped error
    T = poseToMatrix(state[:6])

    # Add estimated joint errors to the robot link
    joint_angle_readings[-(state.shape[0]-6):] += state[6:]
    robot_arm.updateJointAngles(joint_angle_readings)

    # Project points from base to 2D L/R camera image planes
    # Get shaft feature points and lines from FK transform in base frame
    p_b, d_b, _, r = robot_arm.getShaftFeatures()
    # Transform shaft featurepoints from base frame to camera-to-base frame
    p_c = np.dot(np.dot(cam_T_b, T), np.transpose(np.concatenate((p_b, np.ones((p_b.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    
    # Rotate directions from base to camera frame (no translation)
    d_c = np.dot(np.dot(cam_T_b, T)[0:3, 0:3], np.transpose(d_b))
    d_c = np.transpose(d_c)
    
    # Project shaft lines from L and R camera-to-base frames onto 2D camera image plane
    projected_lines = cam.projectShaftLines(p_c, d_c, r, draw_lines = True)

        # Raise error if number of cameras doesn't line up
    if len(line_detections) != len(projected_lines):
        raise ValueError("Length of projected_lines is {} but length of line_detections is {}.\n".format(len(projected_lines), 
                                                                                                            len(line_detections)) \
                        + "Note that these lengths represent the number of cameras being used.")
    
    # Make association between detected and projected & compute probability
    prob = 1
    # len(projected_points) = # of cameras
    # each list in projected points (2x for R/L cameras) is also a list of projected points
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


    # Calculate probability of camera-to-base transform

    prob_l = shaftFeatureObs_SingleCam(detected_lines_l, projected_lines_l, gamma_rho, gamma_theta, rho_thresh, theta_thresh)
    prob_r = shaftFeatureObs_SingleCam(detected_lines_r, projected_lines_r, gamma_rho, gamma_theta, rho_thresh, theta_thresh)
    total_prob = prob_l + prob_r

    return total_prob

def shaftFeatureObs_SingleCam(detected_lines, projected_lines, gamma_rho, gamma_theta, rho_thresh, theta_thresh):
    
    # Maximum association divergence
    max_cost = gamma_rho * rho_thresh + gamma_theta * theta_thresh

    # Check if no projections or detections
    if (detected_lines is None) or (projected_lines is None):
        return 0
    
    rho_detect = detected_lines[:, 0]
    theta_detect = detected_lines[:, 1]
    
    rho_proj= projected_lines[:, 0]
    theta_proj = projected_lines[:, 1]
    
    rho_diffs = np.abs(rho_detect[:, None] - rho_proj) * gamma_rho
    theta_diffs= np.abs(np.fmod(((theta_detect[:, None] - theta_proj) + (2 * np.pi)), (4 * np.pi)) - (2 * np.pi)) * gamma_theta
    C = rho_diffs + theta_diffs

    # Sort by cost (ascending)
    sorted_row_idx, sorted_col_idx = np.unravel_index(np.argsort(C, axis=None), C.shape)
    paired_costs = []
    detected_checked = []
    projected_checked = []

    for i in range(len(sorted_row_idx)):
        row = sorted_row_idx[i]
        col = sorted_col_idx[i]
        if (C[row, col] >= max_cost):
            break
        if (row in detected_checked) or (col in projected_checked):
            continue
        if (rho_diffs[row, col] >= rho_thresh) or (theta_diffs[row, col] >= theta_thresh):
            continue
        paired_costs.append([row, col, C[row, col]])
        detected_checked.append(row)
        projected_checked.append(col)
    
    # Check if any valid paired costs
    if (len(paired_costs) == 0):
        return projected_lines.shape[0] * np.exp(-1 * max_cost)
    else:
        paired_costs = np.asarray(paired_costs)
        prob = (projected_lines.shape[0] - paired_costs.shape[0]) * np.exp(-1 * max_cost) + np.sum(np.exp(paired_costs[:, -1]))

    return prob
