import cv2
import numpy as np
import os
import time
from datetime import datetime
import math

render_folder = ''
footage_folder = ''

# test_person = 'me'
test_person = 'me02'
# test_person = 'marie01'
# test_person = 'marie02'

# footage_file_name = test_person + '_eye_tracked.0001.avi'
footage_file_name = test_person + '.0001.mov'

render_file_name = test_person + '_processed.0002.avi'
# render_file_name = test_person + '_eye_tracked.0002.avi'
render_file_name_compare = test_person + '_processed_full_compare.0002.avi'

is_BGR = False
is_denoise_img = 1
is_improve_img_contrast = 1
is_fix_border = 1

is_show_result = 0
is_show_compare = 0

is_render = 1
is_render_compare = 1
is_stabilize_rotation = 1

# // In frames. The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 100;


def fix_img_border(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    filter = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, filter, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth_trajectory(trajectory, smoothing_radius):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], smoothing_radius)
    return smoothed_trajectory


def improve_img_contrast(frame):
    # improve contrast by using clahe
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame_out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame_out


def img_denoise(frame, is_BGR):
    # adding fast de-noiser
    if is_BGR:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_out = cv2.fastNlMeansDenoisingColored(frame, None, 5, 1, 7, 21)
    return frame_out


def show_debug():
    print('_______________________________________________________________')
    print('Video file: ', footage_folder + footage_file_name)
    print('Video width/height: ', video_width, ':', video_height)
    print('Stabilize Rotation: ', is_stabilize_rotation)
    print('Denoise video: ', is_denoise_img)
    print('Contrast improve: ', is_improve_img_contrast)
    print('')
    print('Show result: ', is_show_result)
    print('Show compare: ', is_show_compare)
    print('Render result: ', is_render)
    if is_render:
        print('Render file: ' + render_folder + render_file_name)
    print('Render compare: ', is_render_compare)
    if is_render_compare:
        print('Render compare file: ' + render_folder + render_file_name_compare)
    print('_______________________________________________________________')


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


def calc_eta(current_frame, frame_count, fps, steps, percent_done, process_name):
    if current_frame < steps:
        fps += fps
    elif current_frame == steps:
        eta_process_sec = (frame_count / (fps / steps)) / steps
        eta_process_min = ((frame_count / (fps / steps)) / steps) / 60

        print('Calculating ... ' + process_name)
        print((datetime.now().strftime("%H:%M:%S")) + ' ETA: ' +
              str(float("{:.2f}".format(eta_process_min))) + 'min ' +
              str(float("{:.2f}".format(eta_process_sec))) + 'sec')
        return True

    # if current_frame > steps:
    #     if int(frame_count * percent_done) == current_frame :
    #
    #         print(str(int(math.ceil(percent_done / 10.0)) * 10) + '% done')
    #         return True


# _________________________________________________________________________________________________

footage_folder, render_folder = set_paths(test_person)

video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')

video_out_compare = cv2.VideoWriter(render_folder + render_file_name_compare,
                                    video_out_codec, video_fps, (2 * video_width, video_height))
video_out_processed = cv2.VideoWriter(render_folder + render_file_name,
                                      video_out_codec, video_fps, (video_width, video_height))

# ______________________________________________________________________________________________________

show_debug()

_, frame = video_cap.read()

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# pre-define transformation-store array
transforms = np.zeros((video_frame_count - 1, 3), np.float32)

is_show_eta = 1
eta_percent = 0
eta_percent_step = 10
# ______________________________________________________________________________________________________
# calc optical flow transformation

for framenum in range(video_frame_count - 2):

    start_time = time.time()

    # detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    success, curr = video_cap.read()
    if not success:
        break

    # convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # sanity check
    assert prev_pts.shape == curr_pts.shape

    # filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # find transformation matrix
    transform_mat = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)

    # print('Frame: ', i, m[0][0,2])

    # extract translation
    dx = transform_mat[0][0, 2]
    dy = transform_mat[0][1, 2]

    # extract rotation if needed
    if not is_stabilize_rotation:
        da = np.arctan2(transform_mat[0][1, 0], transform_mat[0][0, 0]) * 0
    else:
        da = np.arctan2(transform_mat[0][1, 0], transform_mat[0][0, 0])

    # Store transformation
    transforms[framenum] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    # print("Frame: " + str(i) + "/" + str(video_frame_count) + " -  Tracked points : " + str(len(prev_pts)))

    current_fps = 1.0 / (time.time() - start_time)
    calc_eta(framenum, video_frame_count, current_fps, 10, eta_percent, 'Optical flow')

print('Optical flow done ')
print('_______________________________________________________________')
# ______________________________________________________________________________________________________


# compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = smooth_trajectory(trajectory, SMOOTHING_RADIUS)

# calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory

# calculate new transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ______________________________________________________________________________________________________
# calc image processing

for framenum in range(video_frame_count - 2):

    start_time = time.time()

    success, frame = video_cap.read()
    if not success:
        break

    # extract transformations from the new transformation array
    dx = transforms_smooth[framenum, 0]
    dy = transforms_smooth[framenum, 1]
    da = transforms_smooth[framenum, 2]

    # create transformation matrix accordingly to new values
    transform_mat = np.zeros((2, 3), np.float32)
    transform_mat[0, 0] = np.cos(da)
    transform_mat[0, 1] = -np.sin(da)
    transform_mat[1, 0] = np.sin(da)
    transform_mat[1, 1] = np.cos(da)
    transform_mat[0, 2] = dx
    transform_mat[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_processed = cv2.warpAffine(frame, transform_mat, (video_width, video_height))

    # ______________________________________________________________________________________________________

    if is_fix_border:
        frame_processed = fix_img_border(frame_processed)

    if is_improve_img_contrast:
        frame_processed = improve_img_contrast(frame_processed)

    if is_denoise_img:
        frame_processed = img_denoise(frame_processed, is_BGR)

    if is_render_compare:
        frame_out = cv2.hconcat([frame, frame_processed])

    # # If the image is too big, resize it.
    # if (frame_out.shape[1] & gt; 1920):
    #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

    if is_render:
        video_out_processed.write(frame_processed)
    if is_show_result:
        cv2.imshow("stabilization", frame_processed)
    if is_render_compare:
        video_out_compare.write(frame_out)
    if is_show_compare:
        cv2.imshow("Before and after stabilization", frame_out)

    cv2.waitKey(1)

    current_fps = 1.0 / (time.time() - start_time)
    calc_eta(framenum, video_frame_count, current_fps, 2, eta_percent, 'Image processing')

print('Image processing done ')
print('_______________________________________________________________')
