import cv2
import numpy as np

#test_person = 'me'
test_person = 'me02'
#test_person = 'marie01'
#test_person = 'marie02'

render_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'

footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + test_person + '/'

#footage_folder = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + test_person + '/'


footage_file_name = test_person + '_eye_tracked.0001.avi'
#render_file_name = test_person + '_processed.0001.avi'
render_file_name = test_person + '_eye_tracked.0002.avi'
render_file_name_compare = test_person + '_processed_compare_S100.0002.avi'

is_BGR = False
is_denoise_img = 0
is_improve_img_contrast = 0
is_render_compare = 1
is_stabilize_rotation = 0

def fix_img_border(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
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


video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')

video_out_compare = cv2.VideoWriter(render_folder + render_file_name_compare, video_out_codec, video_fps, (2 * video_width, video_height))
video_out_processed = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps, (video_width, video_height))



print('_______________________________________________________________')
print('Video file: ', footage_folder + footage_file_name)
print('Video width/height: ', video_width, ':', video_height)
print('Stabilize Rotation: ', is_stabilize_rotation)
print('Render compare video: ', is_render_compare)
print('Denoise video: ', is_denoise_img)
print('Contrast improve: ', is_improve_img_contrast)
print('_______________________________________________________________')


_, frame = video_cap.read()

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((video_frame_count - 1, 3), np.float32)

for i in range(video_frame_count - 2):

    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
    # Read next frame
    success, curr = video_cap.read()
    if not success:
        break

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix
    m = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)

    # print('Frame: ', i, m[0][0,2])

    # Extract translation
    dx = m[0][0, 2]
    dy = m[0][1, 2]

    # Extract rotation angle
   # da = np.arctan2(m[0][1, 0], m[0][0, 0])

    if not is_stabilize_rotation:
        da = np.arctan2(m[0][1, 0], m[0][0, 0])*0
    else:
        da = np.arctan2(m[0][1, 0], m[0][0, 0])

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    # print("Frame: " + str(i) + "/" + str(video_frame_count) + " -  Tracked points : " + str(len(prev_pts)))

    # cv2.imshow("Frame", prev_gray)
    # cv2.waitKey(1)

# // In frames. The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 100;

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = smooth_trajectory(trajectory, SMOOTHING_RADIUS)

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(video_frame_count - 2):
    # Read next frame
    success, frame = video_cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_processed = cv2.warpAffine(frame, m, (video_width, video_height))

    # Fix border artifacts
    frame_processed = fix_img_border(frame_processed)

    if is_improve_img_contrast:
        # improve contrast
        frame_processed = improve_img_contrast(frame_processed)

    if is_denoise_img:
        # denoise image
        frame_processed = img_denoise(frame_processed, is_BGR)

    if is_render_compare:
        # write the frame to the file
        frame_out = cv2.hconcat([frame, frame_processed])

    # # If the image is too big, resize it.
    # if (frame_out.shape[1] & gt; 1920):
    #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

    #cv2.imshow("Before and after stabilization", frame_processed)
    cv2.waitKey(10)

    if is_render_compare:
        video_out_compare.write(frame_out)
        cv2.imshow("Before and after stabilization", frame_out)

    video_out_processed.write(frame_processed)
