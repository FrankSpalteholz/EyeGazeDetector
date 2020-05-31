import cv2
import numpy as np
import dlib
import xml.etree.ElementTree as ET


# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),  # Nose tip
    (399, 561),  # Chin
    (337, 297),  # Left eye left corner
    (513, 301),  # Right eye right corner
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


calib_filename = 'src/calibData_camL_720p.xml'

camera_matrix_np = load_from_opencv_xml(calib_filename, "KMatCam1")
dist_coeffs_np = np.zeros((4, 1))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera Matrix :\n {0}".format(camera_matrix_np))
print("Distortion Coeffs :\n {0}".format(dist_coeffs_np))



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
