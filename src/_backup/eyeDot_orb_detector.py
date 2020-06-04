import numpy as np
import cv2
from matplotlib import pyplot as plt

#test_person = 'me'
test_person = 'me02'
#test_person = 'marie01'
#test_person = 'marie02'

render_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'
#footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'

footage_still_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + test_person + '/'


footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + test_person + '/'

#footage_file_name = test_person + '_eye_tracked.0003.mov'
footage_file_name = test_person + '.0001.mov'
footage_still_name = test_person + '_triculum.0002.png'
render_file_name = test_person + '_eyeDot_tracked_orb.0001.avi'


video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

print('Input video width: ', video_width)
print('Input video height: ', video_height)

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_orb = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps, (video_width, video_height))

img_to_match = cv2.imread(footage_still_folder + footage_still_name, cv2.IMREAD_GRAYSCALE)



_, frame = video_cap.read()

# Initiate ORB detector
orb = cv2.ORB_create()




for i in range(video_frame_count - 2):
    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (540, 960))

    kp2, des2 = orb.detectAndCompute(gray, None)
    kp1, des1 = orb.detectAndCompute(img_to_match, None)
    # # find the keypoints with ORB
    # kp = orb.detect(gray,None)
    # # compute the descriptors with ORB
    # kp, des = orb.compute(gray, kp)
    # # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matching_result = cv2.drawMatches(img_to_match, kp1, gray, kp2, matches[:150], None, flags=2)

    #cv2.imshow("Img1", gray)
    #cv2.imshow("Img2", img_to_match)
    cv2.imshow("face detection orb", matching_result)
    cv2.waitKey(30)
    video_out_tracked_orb.write(matching_result)