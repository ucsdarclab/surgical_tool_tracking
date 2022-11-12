import numpy as np
import yaml
import cv2

class StereoCamera():
    def __init__(self, cal_file_path, rectify = True, downscale_factor = 2, scale_baseline=1e-3):
        # Load cal file and get all the parameters
        # scale_baseline scales the baseline (i.e. converts units from mm to m!)
        f = open(cal_file_path)
        cal_data = yaml.load(f, Loader=yaml.FullLoader)
        
        self.K1 = np.array(cal_data['K1']['data']).reshape(3,3)
        self.K2 = np.array(cal_data['K2']['data']).reshape(3,3)
        self.D1 = np.array(cal_data['D1']['data'])
        self.D2 = np.array(cal_data['D2']['data'])
        
        self.rotation    = np.array(cal_data['R']['data']).reshape(3,3)
        self.translation = np.array(cal_data['T'])*scale_baseline
        self.T = np.eye(4)
        
        # Downscale stuff
        self.downscale_factor = downscale_factor
        self.img_size = np.array(cal_data['ImageSize'])/self.downscale_factor
        self.img_size = ( int(self.img_size[1]), int(self.img_size[0]) )
        self.K1 = self.K1/self.downscale_factor
        self.K2 = self.K2/self.downscale_factor
        
        self.K1[-1, -1] = 1
        self.K2[-1, -1] = 1
        
        # Prepare undistort and rectification (if desired) here
        if rectify:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K1, self.D1, self.K2, self.D2, 
                                                              self.img_size, self.rotation, self.translation)
            self.left_map1,  self.left_map2   = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1[:,:-1], 
                                                                            self.img_size, cv2.CV_32FC1)
            self.right_map1, self.right_map2  = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2[:,:-1], 
                                                                            self.img_size, cv2.CV_32FC1)
            self.K1 = P1[:,:-1]
            self.K2 = P2[:,:-1]
            
            self.rotation = np.eye(3)
            self.translation = np.linalg.norm(self.translation)*P2[:, -1]/np.linalg.norm(P2[:, -1])

        else:
            self.left_map1, self.left_map2   = cv2.initUndistortRectifyMap(self.K1, self.D1, np.eye(3), self.K1, 
                                                                           self.img_size, cv2.CV_32FC1)
            self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.K2, self.D2, np.eye(3), self.K2, 
                                                                           self.img_size, cv2.CV_32FC1)
        self.T[:3, :3] = self.rotation
        self.T[:3, -1] = self.translation
        
    def processImage(self, left_image, right_image):
        left_image  = cv2.resize(left_image,  self.img_size)
        right_image = cv2.resize(right_image, self.img_size)
        left_image  = cv2.remap(left_image,  self.left_map1,  self.left_map2,  interpolation=cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, self.right_map1, self.right_map2, interpolation=cv2.INTER_LINEAR)
        
        return left_image, right_image

    def projectPoints(self, points):
        # points is Nx3 np array
        points = np.transpose(points)
        projected_point_l = np.transpose(np.dot(self.K1, points/points[-1,:]))[:,:-1]
        
        points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) )
        points_r = np.dot(self.T, points_homogeneous)[:-1, :]
        
        projected_point_r = np.transpose(np.dot(self.K2, points_r/points_r[-1,:]))[:,:-1]
        
        return projected_point_l, projected_point_r
    
    def projectShaftLines_SingleCam(self, points, directions, radii, camera):
        
        # identify L or R camera
        cam_K_matrix = None
        if (camera == 'left'):
            cam_K_matrix = self.K1
        elif (camera == 'right'):
            cam_K_matrix = self.K2
        
        assert(cam_K_matrix is not None)

        # project lines
        projected_lines = []
        
        x0 = points[:, 0]
        y0 = points[:, 1]
        z0 = points[:, 2]

        a = directions[:, 0]
        b = directions[:, 1]
        c = directions[:, 2]

        R = np.asarray(radii) * 1000.0

        alpha1 = np.multiply((np.multiply(-a, a) + 1), x0) - np.multiply(a, np.multiply(b, y0)) - np.multiply(a, np.multiply(c, z0))
        beta1  = np.multiply(-a, np.multiply(b, x0)) + np.multiply((np.multiply(-b, b) + 1), y0) - np.multiply(b, np.multiply(c, z0))
        gamma1 = np.multiply(-a, np.multiply(c, x0)) - np.multiply(b, np.multiply(c, y0)) + np.multiply((np.multiply(-c, c) + 1), z0)

        alpha2 = np.multiply(c, y0) - np.multiply(b, z0)
        beta2  = np.multiply(a, z0) - np.multiply(c, x0)
        gamma2 = np.multiply(b, x0) - np.multiply(a, y0)

        A = np.multiply(x0, x0) + np.multiply(y0, y0) + np.multiply(z0, z0) - np.multiply((np.multiply(a, x0) + np.multiply(b, y0) + np.multiply(c, z0)), (np.multiply(a, x0) + np.multiply(b, y0) + np.multiply(c, z0))) - np.multiply(R, R)

        temp = np.divide(R, np.sqrt(A))

        k1   = np.multiply(alpha1, temp) - alpha2
        k2   = np.multiply(beta1, temp) - beta2
        k3   = np.multiply(gamma1, temp) - gamma2

        F = k1 / cam_K_matrix[0, 0]
        G = k2 / cam_K_matrix[1, 1]
        D = -k3 + F * cam_K_matrix[0, 2] + G * cam_K_matrix[1, 2]

        mask = D < 0
        D = np.where(mask, -D, D)
        G = np.where(mask, -G, G)
        F = np.where(mask, -F, F)
        
        rho = np.divide(D, np.sqrt(np.multiply(F, F) + np.multiply(G, G)))
        theta = np.arctan2(G, F)
        if (rho is not None) and (theta is not None):
            out1 = np.vstack((rho, theta)).T
        else:
            out1 = np.asarray([np.nan, np.nan])

        k1 += 2 * alpha2
        k2 += 2 * beta2
        k3 += 2 * gamma2

        F = k1 / cam_K_matrix[0, 0]
        G = k2 / cam_K_matrix[1, 1]
        D = -k3 + F * cam_K_matrix[0, 2] + G * cam_K_matrix[1, 2]

        mask = D < 0
        D = np.where(mask, -D, D)
        G = np.where(mask, -G, G)
        F = np.where(mask, -F, F)
        
        rho = np.divide(D, np.sqrt(np.multiply(F, F) + np.multiply(G, G)))
        theta = np.arctan2(G, F)
        if (rho is not None) and (theta is not None):
            out2 = np.vstack((rho, theta)).T
        else:
            out2 = np.asarray([np.nan, np.nan])

        projected_lines = np.vstack((out1, out2)) # Nx2 [rho, theta]
        projected_lines = projected_lines[~np.isnan(projected_lines).any(axis=1), :] # filter np.nan
        
        # Return None if no projected lines possible
        if (projected_lines.shape[0] == 0) or (projected_lines.shape[1] != 2):
            return None
        else:
            return projected_lines # Nx2 [rho, theta]

    # Project shaft lines from L/R camera-to-base frames onto 2D camera image plane 
    # points: Nx3 np array of shaft points in L camera-to-base frame (left is default)
    # directions: Nx3 np array of shaft lines in L camera-to-base frame (left is default)
    # radii: Nx1 np array
    def projectShaftLines(self, points, directions, radii):
        
        # Check if points / directions exist
        if (points is None) or (directions is None) or (radii is None):
            return None, None

        # Project lines of 3D cylinder in L camera-to-base frame onto 2D projection on L camera image plane
        points_l = points.copy() #Nx3
        directions_l = directions.copy() #Nx3
        projected_lines_l = self.projectShaftLines_SingleCam(points_l, directions_l, radii, 'left') # Nx2 [rho, theta]

        # Transform L camera-to-base frame points to R camera-to-base frame points
        points = np.transpose(points) # 3xN
        points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) ) # 4xN
        points_r = np.dot(self.T, points_homogeneous)[:-1, :] # 3xN
        points_r = np.transpose(points_r) # Nx3

        # Rotate L camera-to-base frame lines to R camera-to-base frame lines
        directions_r = np.dot(self.T[0:3, 0:3], np.transpose(directions)) # 3xN
        directions_r = np.transpose(directions_r) # Nx3
        projected_lines_r = self.projectShaftLines_SingleCam(points_r, directions_r, radii, 'right') # Nx2 [rho, theta]
        
        return projected_lines_l, projected_lines_r # Nx2 [rho, theta], Nx2 [rho, theta]
    

        
        
