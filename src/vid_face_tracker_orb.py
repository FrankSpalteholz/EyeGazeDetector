import numpy as np
import cv2
from matplotlib import pyplot as plt



video_file = '../render/me/me_out_stabilized.0001.avi'
video_cap = cv2.VideoCapture(video_file)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

print('Input video width: ', video_width)
print('Input video height: ', video_height)

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_orb = cv2.VideoWriter('me_small_out_tracked_orb.0001.avi', video_out_codec, video_fps, (540, 960))


_, frame = video_cap.read()

# Initiate ORB detector
orb = cv2.ORB_create()

for i in range(video_frame_count - 2):
    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (540, 960))

    # find the keypoints with ORB
    kp = orb.detect(gray,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)

    cv2.imshow("face detection orb", img2)
    cv2.waitKey(1)
    video_out_tracked_orb.write(img2)