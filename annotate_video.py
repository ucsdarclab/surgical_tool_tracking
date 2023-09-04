# manual annotations for mouse locations during calibration sequences
# used to create training set for YOLOv5
# click on mouse -> saves dvideo mouse picture + dvideo mouse coordinates
 
import numpy as np
import cv2

localization_file = open('FILE.txt', 'w')

def mouse_event(event, x, y, flags, params):
    
    frame_count = params[0]
    f = params[1]

    # checking for mouse clicks
    if (event == cv2.EVENT_MOUSEMOVE) or (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_RBUTTONDOWN):
        
        text_to_save = str(frame_count) + ',' + str(x) + ',' + str(y) + '\n'
        print(text_to_save)
        f.write(text_to_save)
        
        return True

# Read dvideo and pvideo images
video_cap = cv2.VideoCapture('VIDEO_FILE.mp4')
total_video_frames = video_cap.get(7)

for i in range(1, int(total_video_frames + 1)):

    # get frame
    video_cap.set(1, i)
    # read frame
    video_ret, video_frame = video_cap.read()
    # Display each dvideo_frame
    cv2.imshow('video_frame', video_frame)
    # set mouse callback
    cv2.setMouseCallback('video_frame', mouse_event, param = [i, localization_file])
    # show one frame at a time
    cv2.waitKey(1)