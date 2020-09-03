import numpy as np
import math
import sys


class eye_dot_estimator():

    image_size = np.array([1920, 1080]).astype(int)
    image_center = np.array([image_size[0] / 2, image_size[1] / 2]).astype(int)

    crop_roi_size = np.array([312, 320]).astype(int)

    crop_roi_offset_x1 = 260
    crop_roi_offset_x1 = 52
    crop_roi_offset_y = 160

    landmark_position_eye_rt = np.array([430, 964])
    landmark_position_eye_lf = np.array([650, 964])

    # state 0 = straight eyes
    # state 1 = pointing cam
    # state 2 = pointing mid screen
    # state 3 = pointing left up corner
    # state 4 = pointing left bottom corner
    # state 5 = pointing right bottom corner
    # state 6 = pointing right up corner

    # image pixel space
    calib_positions_image_eye_rt = np.array([[358, 939],
                                       [365, 937],
                                       [362, 959],
                                       [369, 942],
                                       [370, 971],
                                       [352, 972],
                                       [355, 943]])

    calib_positions_image_eye_lf = np.array([[722, 939],
                                       [714, 939],
                                       [712, 958],
                                       [720, 941],
                                       [721, 971],
                                       [706, 972],
                                       [702, 943]])

    # crop roi pixel space
    calib_positions_image_eye_rt = np.array([[187, 136],
                                             [194, 136],
                                             [193, 153],
                                             [200, 138],
                                             [203, 162],
                                             [189, 161],
                                             [185, 137]])

    calib_positions_image_eye_lf = np.array([[125, 138],
                                             [118, 137],
                                             [118, 151],
                                             [127, 137],
                                             [126, 162],
                                             [109, 163],
                                             [107, 137]])

    # simple pythagoras
    def calc_baseline(self, landmark_pos_eye_lf, landmark_pos_eye_rt):
        return math.sqrt(math.abs(landmark_pos_eye_lf[0] - landmark_pos_eye_rt[0]) ** 2 + math.abs(
            landmark_pos_eye_lf[0] - landmark_pos_eye_rt[0]) ** 2)

    def get_roi_origin_in_image(self, side, roi_offset_x1, roi_offset_x2, roi_offset_y, landmark_position):
        roi_origin = np.array([0, 0])

        if side == "R":
            # x position
            roi_origin[0] = landmark_position[0] - roi_offset_x1
            # y position
            roi_origin[1] = landmark_position[1] - roi_offset_y

        if side == "L":
            # x position
            roi_origin[0] = landmark_position[0] - roi_offset_x2
            # y position
            roi_origin[1] = landmark_position[1] - roi_offset_y

        return roi_origin

    def map_eyedot_pos_from_roi_to_image(self, image_size, roi_size, roi_position, position_to_convert):
        converted_position = np.array([0, 0])

        return converted_position

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    estimator = eye_dot_estimator()

    print(estimator.calib_positions_eye_rt[0][1])


if __name__ == "__main__":
    main()
