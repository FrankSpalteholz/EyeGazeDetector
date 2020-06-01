import cv2
import numpy as np

render_folder = '../render/me/'
footage_folder = '../render/me/'

footage_file_name = 'me_small_out_eye_tracked_dlib.0001.avi'
render_file_name = 'me_out_eye_stabilized.0001.avi'
render_file_name_compare = 'me_out_eye_compare.0001.avi'


def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory, smoothing_radius):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], smoothing_radius)
    return smoothed_trajectory


video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')

video_out_compare = cv2.VideoWriter(render_folder + render_file_name_compare, video_out_codec, video_fps, (2 * video_width, video_height))
video_out_stabilized = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps, (video_width, video_height))

is_stabilize_rotation = False

print('_______________________________________________________________')
print('Video file: ', footage_folder + footage_file_name)
print('Video width/height: ', video_width, ':', video_height)
print('Stabilize Rotation: ', is_stabilize_rotation)
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
SMOOTHING_RADIUS = 30;

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)

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
    frame_stabilized = cv2.warpAffine(frame, m, (video_width, video_height))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])

    # # If the image is too big, resize it.
    # if (frame_out.shape[1] & gt; 1920):
    #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

    cv2.imshow("Before and after stabilization", frame_out)
    cv2.waitKey(10)
    video_out_compare.write(frame_out)
    video_out_stabilized.write(frame_stabilized)
