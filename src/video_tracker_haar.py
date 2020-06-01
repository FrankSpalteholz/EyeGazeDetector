import cv2
import numpy as np
import dlib
import xml.etree.ElementTree as ET


render_folder = '../render/me/'
footage_folder = '../render/me/'

footage_file_name = 'me_out_stabilized.0001.avi'
render_file_name = 'me_small_out_tracked_haar.0001.avi'

face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('config/haarcascade_eye.xml')

video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_haar = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps, (540, 960))

print('Input video width: ', video_width)
print('Input video height: ', video_height)

_, frame = video_cap.read()

# Reset stream to first frame
video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



# Write n_frames-1 transformed frames
for i in range(video_frame_count - 2):
    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.equalizeHist(gray, gray)

    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minSize=(500, 500),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                          scaleFactor=1.1,
                                          minSize=(200, 200),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    frame = cv2.resize(frame, (540, 960))

    cv2.imshow("face detection haarcascade", frame)
    cv2.waitKey(1)
    video_out_tracked_haar.write(frame)



