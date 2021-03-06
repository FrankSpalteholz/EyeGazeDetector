import cv2
import dlib
import time
import os

# _________________________________________________________________________________________________________


# test_person = 'me'
test_person = 'me02'
# test_person = 'marie01'
# test_person = 'marie02'

# render_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'
# footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'

# footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + test_person + '/'

#footage_file_name = test_person + '_processed.0001.avi'
#render_file_name = test_person + '_tracked.0001.avi'
#render_eye_file_name = test_person + '_eye_tracked.0001.avi'

footage_file_name = test_person + '_processed_small.0001.mov'
render_file_name = test_person + '_tracked_small.0001.avi'
render_eye_file_name = test_person + '_eye_tracked_small.0001.avi'


predictor_file_path = '../config/shape_predictor_68_face_landmarks.dat'

global_offset_x = 0;
curr_lmpoints_r_eye = []
prev_lmpoints_r_eye = []

offset_x = 50
offset_y = 150
damping = 1.0
is_denoised = 1
is_contrast_improved = 1
is_stabilized = 1
scale = 0.5

is_full_frame_render = 1
is_eye_render = 0
is_full_frame_show = 1
is_eye_frame_show = 0


def display_infos(current_fps):
    font_color = (0, 0, 0)
    cv2.putText(gray, 'fps: ' + str(float("{:.2f}".format(current_fps))), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(gray, 'roi_eye_damping: ' + str(float("{:.2f}".format(damping))), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(gray, 'denoised: ' + 'using NlMean ' + str(is_denoised), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(gray, 'contrast improved: ' + 'using CLAHE ' + str(is_contrast_improved), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(gray, 'stabilized: ' + str(is_stabilized), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)


def set_paths(sub_folder):
    output_path = ''
    input_path = ''
    if os.name == 'nt':
        print("Running system is Win10")
        output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        # output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        input_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\' + sub_folder + r'\\'

    elif os.name == 'posix':
        print("Running system is OSX")
        output_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        # input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + sub_folder + '/'

    return input_path, output_path


footage_folder, render_folder = set_paths(test_person)

video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_dlib = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps,
                                         (int(video_width * scale), int(video_height * scale)))

video_out_eye_tracked_dlib = cv2.VideoWriter()

print('______________________________________________________________')
print('Video to track: ', footage_file_name)
print('Input video width: ', video_width)
print('Input video height: ', video_height)
print('______________________________________________________________')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file_path)

_, frame = video_cap.read()

# Reset stream to first frame
video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

lmarks_left_eye = [42, 43, 44, 45, 46, 47]  # starting from inner corner -> up
lmarks_right_eye = [36, 37, 38, 39, 40, 41]  # starting from outer corner -> up

roi_r_eye_gray = frame

# Write n_frames-1 transformed frames
# for framenum in range((video_frame_count - 2)):
for framenum in range(150):

    start_time = time.time()

    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, ((int(video_width * scale), int(video_height * scale))))

    faces = detector(gray)

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(gray, (x, y), 4, (255, 0, 0), -1)


        if (framenum % 2 != 0):
            #landmarks = predictor(gray, face)

            new_lmpoints_r_eye = []
            for n in range(len(lmarks_right_eye)):
                new_lmpoints_r_eye.append(
                    [landmarks.part(lmarks_right_eye[n]).x, landmarks.part(lmarks_right_eye[n]).y])
                prev_lmpoints_r_eye = curr_lmpoints_r_eye
                curr_lmpoints_r_eye = new_lmpoints_r_eye
        else:
            if (framenum > 1):
                for n in range(len(lmarks_right_eye)):
                    curr_lmpoints_r_eye[n][0] = int(
                        (float(curr_lmpoints_r_eye[n][0]) + float(prev_lmpoints_r_eye[n][0])) / 2)
                    curr_lmpoints_r_eye[n][1] = int(
                        (float(curr_lmpoints_r_eye[n][1]) + float(prev_lmpoints_r_eye[n][1])) / 2)

    # #36==0 outer #39==3 inner
    if (framenum == 1):
        # cv2.imwrite('../footage/me/me.0002.jpg', frame)
        global_offset_x = curr_lmpoints_r_eye[3][0] - curr_lmpoints_r_eye[0][0] + 10

        roi_r_eye_x = int(((float(prev_lmpoints_r_eye[3][0]) + float(curr_lmpoints_r_eye[3][0])) / 2)) + offset_x
        roi_r_eye_y = int(((float(prev_lmpoints_r_eye[3][1]) + float(curr_lmpoints_r_eye[3][1])) / 2)) + offset_y

        roi_r_eye_x2 = roi_r_eye_x - 2 * offset_x - global_offset_x
        roi_r_eye_y2 = roi_r_eye_y - 2 * offset_y

        eye_tracking_rect_size = [roi_r_eye_x - roi_r_eye_x2, roi_r_eye_y - roi_r_eye_y2]

        # print(eye_tracking_rect_size)
        video_out_eye_tracked_dlib = cv2.VideoWriter(render_folder + render_eye_file_name,
                                                     video_out_codec, video_fps,
                                                     (eye_tracking_rect_size[0], eye_tracking_rect_size[1]))

    if framenum > 1:
        roi_r_eye_x += int(((int(((float(prev_lmpoints_r_eye[3][0]) + float(
            curr_lmpoints_r_eye[3][0])) / 2)) + offset_x) - roi_r_eye_x) * damping)
        roi_r_eye_y += int(((int(((float(prev_lmpoints_r_eye[3][1]) + float(
            curr_lmpoints_r_eye[3][1])) / 2)) + offset_y) - roi_r_eye_y) * damping)

        roi_r_eye_x2 = roi_r_eye_x - 2 * offset_x - global_offset_x
        roi_r_eye_y2 = roi_r_eye_y - 2 * offset_y

        cv2.rectangle(gray, (roi_r_eye_x, roi_r_eye_y), (roi_r_eye_x2, roi_r_eye_y2), (255, 0, 0), 1)
        roi_r_eye_gray = gray[roi_r_eye_y2:roi_r_eye_y, roi_r_eye_x2:roi_r_eye_x].copy()

    for npoints in range(len(curr_lmpoints_r_eye)):
        cv2.circle(gray, (curr_lmpoints_r_eye[npoints][0], curr_lmpoints_r_eye[npoints][1]), 2, (255, 0, 0), -1)

    current_fps = 1.0 / (time.time() - start_time)
    display_infos(current_fps)

    if is_eye_render:
        if framenum > 1:
            if is_eye_frame_show:
                cv2.imshow("eye detection dlib", roi_r_eye_gray)
            video_out_eye_tracked_dlib.write(cv2.cvtColor(roi_r_eye_gray, cv2.COLOR_GRAY2BGR))

    if is_full_frame_render:
        if is_full_frame_show:
            cv2.imshow("face detection dlib", gray)
        video_out_tracked_dlib.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    cv2.waitKey(1)
