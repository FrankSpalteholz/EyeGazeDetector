import cv2
import numpy as np
import dlib
import time
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import data_io
import json_io

plt.style.use('seaborn-whitegrid')
from scipy.optimize import curve_fit

# _________________________________________________________________________________________________________


# test_person = 'me'
#est_person = 'me02'
test_person = 'gen'
# test_person = 'marie01'
# test_person = 'marie02'


footage_file_name = test_person + '.0001.mov'
render_file_name = test_person + '_tracked_calib.0001.avi'
render_l_eye_file_name = test_person + '_eye_l_tracked.0001.avi'
render_r_eye_file_name = test_person + '_eye_r_tracked.0001.avi'

footage_folder, render_folder, config_path, data_path = data_io.get_paths(test_person)

print('footage folder:' + footage_folder)
print('render folder:' + render_folder)

predictor_file_path = config_path + 'shape_predictor_68_face_landmarks.dat'

lmarks_left_eye = [42, 43, 44, 45, 46, 47]  # starting from inner corner -> up
lmarks_right_eye = [36, 37, 38, 39, 40, 41]  # starting from outer corner -> up

curr_lmpoints_r_eye_x = []
curr_lmpoints_r_eye_y = []
curr_lmpoints_l_eye_x = []
curr_lmpoints_l_eye_y = []

roi_eye_offset_x = 260
roi_eye_offset_y = 160

roi_eye_damping = 1.0

is_denoised = 1
is_contrast_improved = 1
is_stabilized = 1
scale = 1

is_full_frame_render = 1
is_eye_render = 1
is_full_frame_show = 0
is_eye_frame_show = 1


def display_infos(img, current_fps):
    font_color = (0, 0, 0)
    cv2.putText(img, 'fps: ' + str(float("{:.2f}".format(current_fps))), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(img, 'roi_eye_damping: ' + str(float("{:.2f}".format(roi_eye_damping))), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(img, 'denoised: ' + 'using NlMean ' + str(is_denoised), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(img, 'contrast improved: ' + 'using CLAHE ' + str(is_contrast_improved), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

    cv2.putText(img, 'stabilized: ' + str(is_stabilized), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)


def lin_smooth(data, stop, iterations):
    for it in range(iterations):
        for n in range(stop - 2):
            if n > 1:
                data[n] = ((data[n - 2] + data[n - 1] + data[n] + data[n + 1] + data[n + 2]) / 5)
    return data


def calc_eyes_roi(track_data_l, track_data_r, timestep, roi_offset_x, roi_offset_y):
    roi_list = []

    tmp_list = []
    # roi_r_eye_x \
    tmp_list.append(int(track_data_r[0][timestep] - roi_offset_x))
    # roi_r_eye_y \
    tmp_list.append(int(track_data_r[1][timestep] - roi_offset_y))
    roi_list.append(tmp_list)

    tmp_list = []
    # roi_l_eye_x \
    tmp_list.append(int(track_data_l[0][timestep] + roi_offset_x))
    # roi_l_eye_y \
    tmp_list.append(int(track_data_l[1][timestep] - roi_offset_y))
    roi_list.append(tmp_list)

    tmp_list = []
    # roi_r_eye_x2 \
    tmp_list.append(int(track_data_r[0][timestep] + roi_offset_x / 5))
    # roi_r_eye_y2 \
    tmp_list.append(int(track_data_r[1][timestep] + roi_offset_y))
    roi_list.append(tmp_list)

    tmp_list = []
    # roi_l_eye_x2 \
    tmp_list.append(int(track_data_l[0][timestep] - roi_offset_x / 5))

    # roi_l_eye_y2 \
    tmp_list.append(int(track_data_l[1][timestep] + roi_offset_y))
    roi_list.append(tmp_list)

    return roi_list


def display_track_shapes(img, framenum, data_r_track, data_l_track, roi_eyes_pos):
    cv2.circle(img, (int(data_r_track[0][framenum]), int(data_r_track[1][framenum])), 4, (255, 255, 255), -1)
    cv2.circle(img, (int(data_l_track[0][framenum]), int(data_l_track[1][framenum])), 4, (255, 255, 255), -1)

    cv2.line(img, ((int(data_r_track[0][framenum]), int(data_r_track[1][framenum]))),
             ((int(data_l_track[0][framenum]), int(data_l_track[1][framenum]))), (255, 255, 255), 1)
    cv2.putText(img, 'Baseline: ' + str(int((data_l_track[0][framenum] - data_r_track[0][framenum]) / 2)),
                (int((data_l_track[0][framenum] - data_r_track[0][framenum]) / 4 + data_r_track[0][
                    framenum]),
                 int((data_l_track[1][framenum] - data_r_track[1][framenum]) / 2 + data_l_track[1][
                     framenum] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, 'Baseline: ' + str(int((data_l_track[0][framenum] - data_r_track[0][framenum]) / 2)),
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (roi_eyes_pos[0][0], roi_eyes_pos[0][1]), (roi_eyes_pos[2][0], roi_eyes_pos[2][1]),
                  (255, 255, 255), 1)
    cv2.rectangle(img, (roi_eyes_pos[1][0], roi_eyes_pos[1][1]), (roi_eyes_pos[3][0], roi_eyes_pos[3][1]),
                  (255, 255, 255), 1)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
    video_out_tracked = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps,
                                        (int(video_width * scale), int(video_height * scale)))

    video_out_eye_l_tracked = cv2.VideoWriter()
    video_out_eye_r_tracked = cv2.VideoWriter()

    print('______________________________________________________________')
    print('Video to track: ', footage_file_name)
    print('Input video width: ', video_width)
    print('Input video height: ', video_height)
    print('______________________________________________________________')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file_path)

    _, frame = video_cap.read()
    roi_r_eye_frame = frame
    roi_l_eye_frame = frame

    # reset stream to first frame
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ______________________________________________________________________________________________
    # face landmark detection

    for framenum in range((video_frame_count - 2)):
        # for framenum in range(100):

        start_time = time.time()

        success, frame = video_cap.read()
        if not success:
            break
        frame = cv2.resize(frame, ((int(video_width * scale), int(video_height * scale))))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:

            landmarks = predictor(gray, face)

            curr_lmpoints_r_eye_x.append(landmarks.part(lmarks_right_eye[3]).x)
            curr_lmpoints_r_eye_y.append(landmarks.part(lmarks_right_eye[3]).y)
            curr_lmpoints_l_eye_x.append(landmarks.part(lmarks_left_eye[0]).x)
            curr_lmpoints_l_eye_y.append(landmarks.part(lmarks_left_eye[0]).y)

    # ______________________________________________________________________________________________

    x = curr_lmpoints_r_eye_x.copy()
    y = curr_lmpoints_r_eye_y.copy()

    iterations = 9
    lmark_r_eye_track_x = lin_smooth(curr_lmpoints_r_eye_x, framenum, iterations)
    lmark_r_eye_track_y = lin_smooth(curr_lmpoints_r_eye_y, framenum, iterations)
    lmark_l_eye_track_x = lin_smooth(curr_lmpoints_l_eye_x, framenum, iterations)
    lmark_l_eye_track_y = lin_smooth(curr_lmpoints_l_eye_y, framenum, iterations)

    fig, axs = plt.subplots(2)
    fig.suptitle('x-y motion landmark[3]')
    axs[0].plot(x, 'tab:red', lmark_r_eye_track_x, 'tab:green')
    axs[1].plot(y, 'tab:green', lmark_r_eye_track_y, 'tab:orange')

    plt.show()



    # _____________________________________________________________________________________________
    # SAVING DATA TO JSON
    # _____________________________________________________________________________________________
    # path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/config/data/me02/'

    # json_io.save_np_to_json(np.array([lmark_r_eye_track_x, lmark_r_eye_track_y]), 'array', path + 'data.json')
    #json_io.save_np_to_json(np.array([x, y]), 'array', path + 'data_raw.json')


    # print('clean data:')
    # npdata = json_io.load_np_from_json(path + 'data.json', 'array')
    #
    # print(npdata)
    # print(npdata.dtype)
    # print(npdata.size)
    # print(npdata.shape)
    #
    # print('raw data:')
    # npdata = json_io.load_np_from_json(path + 'data_raw.json', 'array')
    #
    # print(npdata)
    # print(npdata.dtype)
    # print(npdata.size)
    # print(npdata.shape)

    # _____________________________________________________________________________________________

    # Reset stream to first frame
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for framenum in range(video_frame_count - 4):
        # for framenum in range(94):
        # Read next frame
        success, frame = video_cap.read()
        if not success:
            break

        frame = cv2.resize(frame, ((int(video_width * scale), int(video_height * scale))))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # LIN SMOOTH DATA

        roi_eyes_pos = calc_eyes_roi((lmark_l_eye_track_x, lmark_l_eye_track_y), (lmark_r_eye_track_x, lmark_r_eye_track_y),
                                     framenum, roi_eye_offset_x, roi_eye_offset_y)

        display_track_shapes(frame, framenum, (lmark_r_eye_track_x, lmark_r_eye_track_y),
                             (lmark_l_eye_track_x, lmark_l_eye_track_y), roi_eyes_pos)

        # roi_eyes_pos = calc_eyes_roi((curr_lmpoints_l_eye_x, curr_lmpoints_l_eye_y),
        #                              (curr_lmpoints_r_eye_x, curr_lmpoints_r_eye_y),
        #                              framenum, roi_eye_offset_x, roi_eye_offset_y)
        #
        # display_track_shapes(frame, framenum, (curr_lmpoints_r_eye_x, curr_lmpoints_r_eye_y),
        #                      (curr_lmpoints_l_eye_x, curr_lmpoints_l_eye_y), roi_eyes_pos)

        print(roi_eyes_pos)

        roi_r_eye_frame = gray[roi_eyes_pos[0][1]:roi_eyes_pos[2][1], roi_eyes_pos[0][0]:roi_eyes_pos[2][0]].copy()
        roi_l_eye_frame = gray[roi_eyes_pos[1][1]:roi_eyes_pos[3][1], roi_eyes_pos[3][0]:roi_eyes_pos[1][0]].copy()

        eye_rect_size = [abs(roi_eyes_pos[0][0] - roi_eyes_pos[2][0]), abs(roi_eyes_pos[0][1] - roi_eyes_pos[2][1])]


        if framenum == 1:
            video_out_eye_l_tracked = cv2.VideoWriter(render_folder + render_l_eye_file_name,
                                                      video_out_codec, video_fps, (eye_rect_size[0], eye_rect_size[1]))
            video_out_eye_r_tracked = cv2.VideoWriter(render_folder + render_r_eye_file_name,
                                                      video_out_codec, video_fps, (eye_rect_size[0], eye_rect_size[1]))

        # #
        # #     roi_r_eye_x += int(((int(((float(prev_lmpoints_r_eye[3][0]) + float(curr_lmpoints_r_eye[3][0])) / 2)) + roi_eye_offset_x) - roi_r_eye_x) * roi_eye_damping)
        # #     roi_r_eye_y += int(((int(((float(prev_lmpoints_r_eye[3][1]) + float(curr_lmpoints_r_eye[3][1])) / 2)) + roi_eye_offset_y) - roi_r_eye_y) * roi_eye_damping)
        # #
        #
        #

        # current_fps = 1.0 / (time.time() - start_time)
        # display_infos(current_fps)

        if is_eye_render:
            if framenum > 1:
                if is_eye_frame_show:
                    cv2.imshow("eye detection dlib", cv2.hconcat([roi_r_eye_frame, roi_l_eye_frame]))
                    # cv2.imshow("eye detection dlib", roi_l_eye_frame)
                video_out_eye_r_tracked.write(cv2.cvtColor(roi_r_eye_frame, cv2.COLOR_GRAY2BGR))
                video_out_eye_l_tracked.write(cv2.cvtColor(roi_l_eye_frame, cv2.COLOR_GRAY2BGR))

        if is_full_frame_render:
            if is_full_frame_show:
                cv2.imshow("face detection dlib", frame)
            video_out_tracked.write(frame)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()
