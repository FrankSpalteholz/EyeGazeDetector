import cv2
import numpy as np


render_folder = '../render/me/'
footage_folder = '../render/me/'

footage_file_name = 'me_out_eye_stabilized.0001.avi'
render_file_name = 'me_small_out_eyeDot_tracked.0001.avi'


cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))


video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_dlib = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps, (video_width, video_height))

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    roi = frame
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)


    ret , threshold = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
    #
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    color = (0, 225, 255)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        #cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.circle(roi, (x + int(w / 2), y + int(h / 2)), 10, color, 1)
        cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), color, 1)
        cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), color, 1)
        break

    #cv2.imshow("Threshold", threshold)
    #cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    video_out_tracked_dlib.write(roi)

    key = cv2.waitKey(30)

    if key == 27:
        break

cv2.destroyAllWindows()