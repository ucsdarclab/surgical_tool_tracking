# manual annotations for mouse locations during calibration sequences
# used to create training set for YOLOv5
# click on mouse -> saves dvideo mouse picture + dvideo mouse coordinates

import pandas as pd
import numpy as np
import cv2
import csv

# load main dataframe
merged_df = np.load('Demo5/merged_df.pkl', allow_pickle = True)

# dvideo calibration frames
# Demo3
# 6800-8160
# 13250 - 15300
# 19800-21200
# Demo4
# 5605-6900
# 16020-17045
# 28030-28995
# Demo5
# 4900-6100
# 18100-19380
# 49000-50000
start_cal1 = 4900
end_cal1 = 6100
start_cal2 = 18100
end_cal2 = 19380
start_cal3 = 49000
end_cal3 = 50000

# create list of calibration frames
dvideo_frames = []
pvideo_frames = []

temp = merged_df.loc[(merged_df['dvideo_frame'] >= start_cal1) & (merged_df['dvideo_frame'] <= end_cal1)]
dvideo_frames.extend(list(temp['dvideo_frame']))
pvideo_frames.extend(list(temp['pvideo_frame']))
temp = merged_df.loc[(merged_df['dvideo_frame'] >= start_cal2) & (merged_df['dvideo_frame'] <= end_cal2)]
dvideo_frames.extend(list(temp['dvideo_frame']))
pvideo_frames.extend(list(temp['pvideo_frame']))
temp = merged_df.loc[(merged_df['dvideo_frame'] >= start_cal3) & (merged_df['dvideo_frame'] <= end_cal3)]
dvideo_frames.extend(list(temp['dvideo_frame']))
pvideo_frames.extend(list(temp['pvideo_frame']))

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
 
    # checking for mouse clicks
    if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_RBUTTONDOWN):

        dvideo_frame_counter = params[0]
        dvideo_image = params[1]
        pvideo_frame_counter = params[2]
        pvideo_image = params[3]
        
        # dvideo image filename for saving
        dvideo_save_filename = str(dvideo_frame_counter)
        pvideo_save_filename = str(pvideo_frame_counter)
        # print dvideo frame number, coordinates of mouse click, pvideo frame number
        print(dvideo_save_filename, ' ', x, ' ', y, pvideo_save_filename)
        # write image labels to csv
        dvideo_save_filename = '2021_11_10_13_13_49_dvideo_' + dvideo_save_filename + '.png'
        pvideo_save_filename = '2021_11_10_13_13_58_pupil_annotated_' + pvideo_save_filename + '.png'
        dvideo_save_text = [dvideo_save_filename, x, y]
        pvideo_save_text = [pvideo_save_filename, x, y]
        with open("mouse_net/mouse_labels.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(dvideo_save_text)
        with open("transform_net/transform_labels.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(pvideo_save_text)

        # save image files
        dvideo_save_filename = 'mouse_net/pictures/' + dvideo_save_filename
        pvideo_save_filename = 'transform_net/pictures/' + pvideo_save_filename
        # write images to file
        cv2.imwrite(dvideo_save_filename, dvideo_image)
        cv2.imwrite(pvideo_save_filename, pvideo_image)


# Read dvideo and pvideo images
dvideo_cap = cv2.VideoCapture('/Users/christopher/Documents/EEGHRV/Demo5/2021_11_10_13_13_49_dvideo.mp4')
pvideo_cap = cv2.VideoCapture('/Users/christopher/Documents/EEGHRV/Demo5/2021_11_10_13_13_58_pupil_annotated.avi')
total_dvideo_frames = dvideo_cap.get(7)
total_pvideo_frames = pvideo_cap.get(7)

for i in range(len(dvideo_frames)):

    # frame counter
    dvideo_frame_number = int(dvideo_frames[i])
    pvideo_frame_number = int(pvideo_frames[i])
    # get frame
    dvideo_cap.set(1, dvideo_frame_number)
    pvideo_cap.set(1, pvideo_frame_number)
    # read frame
    dvideo_ret, dvideo_frame = dvideo_cap.read()
    pvideo_ret, pvideo_frame = pvideo_cap.read()
    # Display each dvideo_frame
    cv2.imshow('dvideo_frame', dvideo_frame)
    # set mouse callback
    cv2.setMouseCallback('dvideo_frame', click_event, param = [dvideo_frame_number, dvideo_frame, pvideo_frame_number, pvideo_frame])
    # show one frame at a time
    cv2.waitKey(0)