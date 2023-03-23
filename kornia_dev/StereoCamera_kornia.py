import numpy as np
import yaml
import cv2
import math

class StereoCamera():
    def __init__(self, cal_file_path, rectify = True, orig_ref_dims = None, crop_ref_dims = None, downscale_factor = 2, scale_baseline=1e-3):
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

        # re-center for crop
        if (orig_ref_dims is not None) and (crop_ref_dims is not None):
                height, width = orig_ref_dims[0], orig_ref_dims[1]
                mid_y, mid_x = int(height / 2), int(width / 2)

                crop_height, crop_width = int(crop_ref_dims[0] / 2), int(crop_ref_dims[1] / 2)
                y_offset = mid_y - crop_height
                x_offset = mid_x - crop_width

                # cx' = cx - x_offset
                self.K1[0, -1] = self.K1[0, -1] - x_offset
                self.K2[0, -1] = self.K2[0, -1] - x_offset

                # cy' = cy - y_offset
                self.K1[1, -1] = self.K1[1, -1] - y_offset
                self.K2[1, -1] = self.K2[1, -1] - y_offset

        
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
    
    # https://github.com/ucsdarclab/dvrk_particle_filter/blob/master/src/stereo_camera.cpp#L252
    def projectShaftLines_SingleCam(self, points, directions, radii, camera):

        print('in projectShaftLines_SingleCam')
        print('points: {}'.format(points))
        print('points.shape: {}'.format(points.shape))
        print('directions: {}'.format(directions))
        print('directions.shape: {}'.format(directions.shape))
        print('radii: {}'.format(radii))
        print('camera: {}'.format(camera))
        
        # identify L or R camera
        cam_K_matrix = None
        if (camera == 'left'):
            cam_K_matrix = self.K1
        elif (camera == 'right'):
            cam_K_matrix = self.K2   
        assert(cam_K_matrix is not None)

        print('cam_K_matrix: {}'.format(cam_K_matrix))

        assert(len(points) == len(directions) == len(radii))
        projected_lines = []
        for i in range(points.shape[0]):
            
            x0 = points[i, 0]
            print('x0: {}'.format(x0))
            y0 = points[i, 1]
            print('y0: {}'.format(y0))
            z0 = points[i, 2]
            print('z0: {}'.format(z0))

            a = directions[i, 0]
            print('a: {}'.format(a))
            b = directions[i, 1]
            print('b: {}'.format(b))
            c = directions[i, 2]
            print('c: {}'.format(c))

            # are these units correct? makes A < 0
            R = radii[i]
            print('R: {}'.format(R))

            alpha1 = (1 - a * a) * x0 - a * b * y0 - a * c * z0
            beta1  = -a * b * x0 + (1 - b * b) * y0 - b * c * z0
            gamma1 = -a * c * x0 - b * c * y0 + (1 - c * c) * z0

            alpha2 = c * y0 - b * z0
            beta2  = a * z0 - c * x0
            gamma2 = b * x0 - a * y0

            A = x0 * x0 + y0 * y0 + z0 * z0 - (a * x0 + b * y0 + c * z0) * (a * x0 + b * y0 + c * z0) - R * R
            print('A: {}'.format(A))

            if (A <= 0):
                continue

            temp = R / np.sqrt(A)

            k1   = (alpha1 * temp - alpha2)
            k2   = (beta1 * temp - beta2)
            k3   = (gamma1 * temp - gamma2)

            F = k1 / cam_K_matrix[0, 0]
            G = k2 / cam_K_matrix[1, 1]
            D = -k3 + F * cam_K_matrix[0, 2] + G * cam_K_matrix[1, 2]

            if (D < 0):
                D *= -1
                G *= -1
                F *= -1      
            
            rho = D / np.sqrt(F * F + G * G)
            theta = math.atan2(G, F)
            projected_lines.append([rho, theta])

            k1 += 2 * alpha2
            k2 += 2 * beta2
            k3 += 2 * gamma2

            F = k1 / cam_K_matrix[0, 0]
            G = k2 / cam_K_matrix[1, 1]
            D = -k3 + F * cam_K_matrix[0, 2] + G * cam_K_matrix[1, 2]

            if (D < 0):
                D *= -1
                G *= -1
                F *= -1
            
            rho = D / np.sqrt(F * F + G * G)
            theta = math.atan2(G, F) # radians

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

            projected_lines.append([rho, theta])
            
        projected_lines = np.asarray(projected_lines)
        print('projected_lines from projectShaftLines_SingleCam: {}'.format(projected_lines))
        return projected_lines # Nx2 [rho, theta] # theta in radians

    # Project shaft lines from L/R camera-to-base frames onto 2D camera image plane 
    # points: Nx3 np array of shaft points in L camera-to-base frame (left is default)
    # directions: Nx3 np array of shaft lines in L camera-to-base frame (left is default)
    # radii: Nx1 np array
    def projectShaftLines(self, points, directions, radii):

        print('in projectShaftLines')
        print('points: {}'.format(points))
        print('points.shape: {}'.format(points.shape))
        print('directions: {}'.format(directions))
        print('directions.shape: {}'.format(directions.shape))
        print('radii: {}'.format(radii))
        
        # Check if points / directions exist
        if (points is None) or (directions is None) or (radii is None):
            return None, None

        # Project lines of 3D cylinder in L camera-to-base frame onto 2D projection on L camera image plane
        points_l = points.copy() #Nx3
        print('points_l: {}'.format(points_l))
        print('points_l.shape: {}'.format(points_l.shape))

        directions_l = directions.copy() #Nx3
        print('directions_l: {}'.format(directions_l))
        print('directions_l.shape: {}'.format(directions_l.shape))

        projected_lines_l = self.projectShaftLines_SingleCam(points_l, directions_l, radii, 'left') # Nx2 [rho, theta]

        # Transform L camera-to-base frame points to R camera-to-base frame points
        points = np.transpose(points) # 3xN
        points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) ) # 4xN
        points_r = np.dot(self.T, points_homogeneous)[:-1, :] # 3xN
        points_r = np.transpose(points_r) # Nx3

        print('points_r.shape: {}'.format(points_r.shape))

        # Rotate L camera-to-base frame lines to R camera-to-base frame lines
        directions_r = np.dot(self.T[0:3, 0:3], np.transpose(directions)) # 3xN
        directions_r = np.transpose(directions_r) # Nx3
        projected_lines_r = self.projectShaftLines_SingleCam(points_r, directions_r, radii, 'right') # Nx2 [rho, theta]
        
        print('returning projected lines from projectShaftLines')
        print('projected_lines_l: {}'.format(projected_lines_l))
        print('projected_lines_r: {}'.format(projected_lines_r))
        return projected_lines_l, projected_lines_r # Nx2 [rho, theta], Nx2 [rho, theta]