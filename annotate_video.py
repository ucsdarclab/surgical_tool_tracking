# manual annotations for mouse locations during calibration sequences
# used to create training set for YOLOv5
# click on mouse -> saves dvideo mouse picture + dvideo mouse coordinates
 
import numpy as np
import cv2
import csv

# load main dataframe

# function to display the coordinates of
# of the points clicked on the image
def mouse_event(event, x, y, flags, params):
    
    frame_counter = int(params[0])

    # checking for mouse clicks
    if (event == cv2.EVENT_MOUSEMOVE) or (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_RBUTTONDOWN):
        
        text_to_save = [frame_counter, x, y]
        print(text_to_save)
        with open("keypoint_labels.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(text_to_save)

# Read dvideo and pvideo images
video_cap = cv2.VideoCapture('../journal_dataset/right_video.mp4')
total_video_frames = video_cap.get(7)

for i in range(1, int(total_video_frames + 1)):

    # get frame
    video_cap.set(1, i)
    # read frame
    video_ret, video_frame = video_cap.read()
    # Display each dvideo_frame
    cv2.imshow('video_frame', video_frame)
    print(i)
    # set mouse callback
    cv2.setMouseCallback('video_frame', mouse_event, param = [i])
    # show one frame at a time
    cv2.waitKey(0)