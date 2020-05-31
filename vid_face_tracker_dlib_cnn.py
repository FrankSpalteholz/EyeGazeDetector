import cv2
import numpy as np
import dlib
import time
import xml.etree.ElementTree as ET



video_file = 'render/me/me_out_stabilized.0001.avi'
video_cap = cv2.VideoCapture(video_file)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_dlib = cv2.VideoWriter('render/me/me_small_out_tracked_damped_dlib.0001.avi', video_out_codec, video_fps, (540, 960))
video_out_eye_tracked_dlib = cv2.VideoWriter()

print('______________________________________________________________')
print('Input video width: ', video_width)
print('Input video height: ', video_height)
print('______________________________________________________________')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

_, frame = video_cap.read()

# Reset stream to first frame
video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


lmarks_left_eye = [42,43,44,45,46,47] # starting from inner corner -> up
lmarks_right_eye = [36,37,38,39,40,41] # starting from outer corner -> up


global_offset_x = 0;
curr_lmpoints_r_eye = []
prev_lmpoints_r_eye = []
roi_r_eye_gray = frame
offset_x = 40
offset_y = 80

# Write n_frames-1 transformed frames
#for framenum in range((video_frame_count - 2)):
for framenum in range(200):

    # start time of the loop
    start_time = time.time()

    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (540, 960))

    # improve contrast by using clahe
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame_contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # adding fast de-noiser
    frame_contrast = cv2.cvtColor(frame_contrast, cv2.COLOR_BGR2RGB)
    frame_denoise = cv2.fastNlMeansDenoisingColored(frame_contrast, None, 5, 1, 7, 21)
    frame = frame_denoise

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        if(framenum % 2 != 0):
            landmarks = predictor(gray, face)

            new_lmpoints_r_eye = []
            for n in range(len(lmarks_right_eye)):
                new_lmpoints_r_eye.append([landmarks.part(lmarks_right_eye[n]).x, landmarks.part(lmarks_right_eye[n]).y])
                prev_lmpoints_r_eye = curr_lmpoints_r_eye
                curr_lmpoints_r_eye = new_lmpoints_r_eye
        else:
            if(framenum > 1):
                for n in range(len(lmarks_right_eye)):
                    curr_lmpoints_r_eye[n][0] = int((float(curr_lmpoints_r_eye[n][0]) + float(prev_lmpoints_r_eye[n][0]))/2)
                    curr_lmpoints_r_eye[n][1] = int((float(curr_lmpoints_r_eye[n][1]) + float(prev_lmpoints_r_eye[n][1]))/2)



    # #36==0 outer #39==3 inner
    if(framenum==1):
        cv2.imwrite('footage/me/me.0002.jpg', frame)
        global_offset_x = curr_lmpoints_r_eye[3][0] - curr_lmpoints_r_eye[0][0] + 10

        roi_r_eye_x = int(((float(prev_lmpoints_r_eye[3][0]) + float(curr_lmpoints_r_eye[3][0]))/2)) + offset_x
        roi_r_eye_y = int(((float(prev_lmpoints_r_eye[3][1]) + float(curr_lmpoints_r_eye[3][1]))/2)) + offset_y


        roi_r_eye_x2 = roi_r_eye_x - offset_x - global_offset_x
        roi_r_eye_y2 = roi_r_eye_y - offset_y

        eye_tracking_rect_size = [roi_r_eye_x - roi_r_eye_x2, roi_r_eye_y - roi_r_eye_y2]

        video_out_eye_tracked_dlib = cv2.VideoWriter('render/me/me_small_out_eye_tracked_dlib.0001.avi',
                                                     video_out_codec, video_fps, (eye_tracking_rect_size[0], eye_tracking_rect_size[1]))

    if framenum > 1:

        damping = 0.5
        roi_r_eye_x += int(((int(((float(prev_lmpoints_r_eye[3][0]) + float(curr_lmpoints_r_eye[3][0])) / 2)) + offset_x) - roi_r_eye_x) * damping)
        roi_r_eye_y += int(((int(((float(prev_lmpoints_r_eye[3][1]) + float(curr_lmpoints_r_eye[3][1])) / 2)) + offset_y) - roi_r_eye_y) * damping)

        roi_r_eye_x2 = roi_r_eye_x - 2*offset_x - global_offset_x
        roi_r_eye_y2 = roi_r_eye_y - 2*offset_y

        cv2.rectangle(gray, (roi_r_eye_x, roi_r_eye_y), (roi_r_eye_x2, roi_r_eye_y2), (255, 0, 0), 1)
        roi_r_eye_gray = gray[ roi_r_eye_y2:roi_r_eye_y,roi_r_eye_x2:roi_r_eye_x].copy()

    for npoints in range(len(curr_lmpoints_r_eye)):
        cv2.circle(gray, (curr_lmpoints_r_eye[npoints][0], curr_lmpoints_r_eye[npoints][1]), 2, (255, 0, 0), -1)

    cv2.putText(gray, 'fps: ' + str(float("{:.2f}".format(1.0 / (time.time() - start_time)))), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    cv2.imshow("face detection dlib", gray)
    # if framenum > 1:
    #     cv2.imshow("face detection dlib", roi_r_eye_gray)
    #     #video_out_eye_tracked_dlib.write(cv2.cvtColor(roi_r_eye_gray, cv2.COLOR_GRAY2BGR))

    cv2.waitKey(1)
    video_out_tracked_dlib.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))



