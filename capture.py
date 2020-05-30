import cv2
import numpy as np
import dlib
import xml.etree.ElementTree as ET

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),  # Nose tip
    (399, 561),  # Chin
    (337, 297),  # Left eye left corner
    (513, 301),  # Right eye right corne
    (345, 465),  # Left Mouth corner
    (453, 469)  # Right mouth corner
], dtype="double")


def load_from_opencv_xml(filename, elementname, dtype='float32'):
    try:
        tree = ET.parse(filename)
        rows = int(tree.find(elementname).find('rows').text)
        cols = int(tree.find(elementname).find('cols').text)
        return np.fromstring(tree.find(elementname).find('data').text, dtype, count=rows * cols, sep=' ').reshape(
            (rows, cols))
    except Exception as e:
        print(e)
        return None


def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 2)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 2)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 2)
    return img


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


video_file = 'me_small.0001.mov'
calib_filename = 'calibData_camL_720p.xml'

camera_matrix_np = load_from_opencv_xml(calib_filename, "KMatCam1")
dist_coeffs_np = np.zeros((4, 1))

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera Matrix :\n {0}".format(camera_matrix_np))
print("Distortion Coeffs :\n {0}".format(dist_coeffs_np))

video_cap = cv2.VideoCapture(video_file)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_compare = cv2.VideoWriter('me_small_out_compare.0001.avi', video_out_codec, video_fps, (2 * video_width, video_height))
video_out_stabilized = cv2.VideoWriter('me_small_out_stabilized.0001.avi', video_out_codec, video_fps, (video_width, video_height))

print('Input video width: ', video_width)
print('Input video height: ', video_height)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
    da = np.arctan2(m[0][1, 0], m[0][0, 0])

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    # print("Frame: " + str(i) + "/" + str(video_frame_count) + " -  Tracked points : " + str(len(prev_pts)))

    # cv2.imshow("Frame", prev_gray)
    # cv2.waitKey(1)

# # // In frames. The larger the more stable the video, but less reactive to sudden panning
# SMOOTHING_RADIUS = 30;
#
# # Compute trajectory using cumulative sum of transformations
# trajectory = np.cumsum(transforms, axis=0)
#
# smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)
#
# # Calculate difference in smoothed_trajectory and trajectory
# difference = smoothed_trajectory - trajectory
#
# # Calculate newer transformation array
# transforms_smooth = transforms + difference
#
# # Reset stream to first frame
# video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # Write n_frames-1 transformed frames
# for i in range(video_frame_count - 2):
#     # Read next frame
#     success, frame = video_cap.read()
#     if not success:
#         break
#
#     # Extract transformations from the new transformation array
#     dx = transforms_smooth[i, 0]
#     dy = transforms_smooth[i, 1]
#     da = transforms_smooth[i, 2]
#
#     # Reconstruct transformation matrix accordingly to new values
#     m = np.zeros((2, 3), np.float32)
#     m[0, 0] = np.cos(da)
#     m[0, 1] = -np.sin(da)
#     m[1, 0] = np.sin(da)
#     m[1, 1] = np.cos(da)
#     m[0, 2] = dx
#     m[1, 2] = dy

    # # Apply affine wrapping to the given frame
    # frame_stabilized = cv2.warpAffine(frame, m, (video_width, video_height))
    #
    # # Fix border artifacts
    # # frame_stabilized = fixBorder(frame_stabilized)
    #
    # # Write the frame to the file
    # frame_out = cv2.hconcat([frame, frame_stabilized])
    #
    # # # If the image is too big, resize it.
    # # if (frame_out.shape[1] & gt; 1920):
    # #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));
    #
    # cv2.imshow("Before and after stabilization", frame_out)
    # cv2.waitKey(10)
    # video_out_compare.write(frame_out)
    # video_out_stabilized.write(frame_stabilized)

while True:

    _, frame = cap.read()



    #==========================================================================================================
    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        image_points[0][0] = landmarks.part(33).x; image_points[0][1] = landmarks.part(33).y;
        image_points[1][0] = landmarks.part(8).x; image_points[1][1] = landmarks.part(8).y;
        image_points[2][0] = landmarks.part(45).x; image_points[2][1] = landmarks.part(45).y;
        image_points[3][0] = landmarks.part(36).x; image_points[3][1] = landmarks.part(36).y;
        image_points[4][0] = landmarks.part(54).x; image_points[4][1] = landmarks.part(54).y;
        image_points[5][0] = landmarks.part(48).x; image_points[5][1] = landmarks.part(48).y;

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix_np,
                                                                  dist_coeffs_np, flags=cv2.SOLVEPNP_ITERATIVE)

    #print("Rotation Vector:\n {0}".format(rotation_vector))
    #print("Translation Vector:\n {0}".format(translation_vector))

    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)


    draw_axis(frame, rotation_vector, translation_vector, camera_matrix_np)

    #==========================================================================================================

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
