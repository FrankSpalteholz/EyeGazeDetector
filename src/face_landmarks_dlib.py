import cv2
import numpy as np
import dlib
import data_io

test_person = 'gen'
footage_file_name = test_person + '.0001.mov'
render_file_name =  test_person + '_tracked.0001.avi'
scale = 1.0


# test_person = 'me02'
# footage_file_name = test_person + '_small_processed.0002.avi'
# render_file_name =  test_person + '_tracked.0001.avi'
#scale = 1

footage_folder, render_folder, config_path, data_path = data_io.get_paths(test_person)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("config/shape_predictor_68_face_landmarks.dat")

video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

print(video_width, video_height)

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps,
                                        (int(video_width * scale), int(video_height * scale)))

for framenum in range(video_frame_count):
    success, frame = video_cap.read()

    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (0, 50, 250), -1)


    cv2.imshow("Frame", frame)

    video_out_tracked.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break