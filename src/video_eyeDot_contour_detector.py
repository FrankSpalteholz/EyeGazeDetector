import cv2
import os
import numpy as np

#draw_mobile_grid(roi, int(58*scale), int(104*scale), 1.0, (int(58*scale), int(135*scale)), 7, 4, True, color_traj, 1)

def draw_trajectory(img, trajectory, color_traj, size, circle_type):
    for pos in trajectory:
        #overlay = roi.copy()
        #alpha = 0.4  # Transparency factor.
        cv2.circle(img, (pos[0], pos[1]), size, color_traj, circle_type)
        # Following line overlays transparent rectangle over the image
        #roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        #cv2.circle(roi, (pos[0], pos[1]), 2, color_traj, -1)

def draw_mobile_grid(img, width, height, scale, pos, w_lines, h_lines, draw_center_lines, color, line_width):

    line_spacing = [0, 0]
    mid = [0, 0]
    line_spacing[0] = int((height/(w_lines + 1)))
    line_spacing[1] = int((width/(h_lines + .8)))
    cam_pos_x = int(width * 0.31)
    cam_pos_y = int(height * 0.077)
    mid[0] = int(pos[0] + (width/2))
    mid[1] = int(pos[1] + (height/2))

    #cv2.circle(roi, (curr_pos[0], curr_pos[1]), 10, color, 1)
    #cv2.line(roi, (curr_pos[0], 0), (curr_pos[0], rows), color, 1)

    cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + width, pos[1] + height), color, line_width)

    if draw_center_lines:
        cv2.line(img, (mid[0], pos[1]), (mid[0], pos[1] + height), color, line_width )
        cv2.line(img, (pos[0], mid[1]), (pos[0] + width, mid[1]), color, line_width )

    cv2.circle(img, (cam_pos_x + pos[0], pos[1] - cam_pos_y), 4, color, line_width)

    for n in range(int(w_lines/2) + 1):

        cv2.line(img,
                 (pos[0], mid[1] + int(line_spacing[0] * n)),
                 (pos[0] + width, mid[1] + int(line_spacing[0] * n)),
                 color, line_width)
        cv2.line(img,
                 (pos[0], mid[1] - int(line_spacing[0] * n)),
                 (pos[0] + width, mid[1] - int(line_spacing[0] * n)),
                 color, line_width)

    for n in range(int(h_lines)):
            cv2.line(img,
                (int(pos[0] + line_spacing[1] * (n+1)), pos[1]),
                (int(pos[0] + line_spacing[1] * (n+1)), pos[1] + height),
                color, line_width)

def draw_info(img, trajectory, traj_proj, framenum, color):

    cv2.putText(roi, '#' + str(framenum), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)

    check_frame1 = 79
    check_frame2 = 134
    check_frame3 = 155
    check_frame4 = 214
    check_frame5 = 72

    if framenum > check_frame1:
        cv2.circle(img, (trajectory[check_frame1][0], trajectory[check_frame1][1]), 3, (0,0,255), -1)
        cv2.circle(img, (traj_proj[check_frame1-1][0], traj_proj[check_frame1-1][1]), 3, (0,0,255), -1)
        cv2.putText(img, 'Corner 1: ' + str(trajectory[check_frame1][0]) + ':' + str(trajectory[check_frame1][1]), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if framenum > check_frame2:
        cv2.circle(img, (trajectory[check_frame2][0], trajectory[check_frame2][1]), 3, (0,0,255), -1)
        cv2.circle(img, (traj_proj[check_frame2-1][0], traj_proj[check_frame2-1][1]), 3, (0,0,255), -1)
        cv2.putText(img, 'Corner 2: ' + str(trajectory[check_frame2][0]) + ':' + str(trajectory[check_frame2][1]), (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if framenum > check_frame3:
        cv2.circle(img, (trajectory[check_frame3][0], trajectory[check_frame3][1]), 3, (0,0,255), -1)
        cv2.circle(img, (traj_proj[check_frame3-1][0], traj_proj[check_frame3-1][1]), 3, (0,0,255), -1)
        cv2.putText(img, 'Corner 3: ' + str(trajectory[check_frame3][0]) + ':' + str(trajectory[check_frame3][1]), (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if framenum > check_frame4:
        cv2.circle(img, (trajectory[check_frame4][0], trajectory[check_frame4][1]), 3, (0,0,255), -1)
        cv2.circle(img, (traj_proj[check_frame4-1][0], traj_proj[check_frame4-1][1]), 3, (0,0,255), -1)
        cv2.putText(img, 'Corner 4: ' + str(trajectory[check_frame4][0]) + ':' + str(trajectory[check_frame4][1]), (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if framenum > check_frame5:
        cv2.circle(img, (trajectory[check_frame5][0], trajectory[check_frame5][1]), 3, (0,0,255), -1)
        cv2.circle(img, (traj_proj[check_frame5-1][0], traj_proj[check_frame5-1][1]), 3, (0,0,255), -1)
        cv2.putText(img, 'Middle: ' + str(trajectory[check_frame5][0]) + ':' + str(trajectory[check_frame5][1]), (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

def set_paths(sub_folder):
    output_path = ''
    input_path = ''
    if os.name == 'nt':
        print("Running system is Win10")
        output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        # output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        input_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'

    elif os.name == 'posix':
        print("Running system is OSX")
        output_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        # input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + sub_folder + '/'

    return input_path, output_path

#______________________________________________________________________________________________________

#test_person = 'me'
test_person = 'me02'
#test_person = 'marie01'
#test_person = 'marie02'


footage_file_name = test_person + '_eye_tracked.0003.avi'
render_file_name = test_person + '_eyeDot_tracked.0003.avi'

footage_folder, render_folder = set_paths(test_person)

video_cap = cv2.VideoCapture(footage_folder + footage_file_name)

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

video_scale = 2.0

video_out_codec = cv2.VideoWriter_fourcc(*'MJPG')
video_out_tracked_dlib = cv2.VideoWriter(render_folder + render_file_name, video_out_codec, video_fps,
                                         (int(video_width * video_scale), int(video_height * video_scale)))

video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))



prev_pos = [0, 0]
curr_pos = [0, 0]
trajectory = []
trajectory_proj = []
corners = []


#______________________________________________________________________________________________________


for framenum in range((video_frame_count - 2)):
    ret, frame = video_cap.read()
    if ret is False:
        break

    frame = cv2.resize(frame, ((int(video_width * video_scale), int(video_height * video_scale))))

    roi = frame
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #gray_roi = cv2.equalizeHist(gray_roi)


    ret , threshold = cv2.threshold(gray_roi, 20, 255, cv2.THRESH_BINARY_INV)
    #
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    damping = 1.0
    color = (0, 225, 255)
    color_traj = (255, 255, 255)

    new_pos = [0, 0]

    grid_x_pos = 450
    grid_y_pos = 50

    traj_offset_x = 315;
    traj_offset_y = 298;


    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        #cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 1)

        prev_pos[0] = curr_pos[0]
        prev_pos[1] = curr_pos[1]
        new_pos[0] =  x + int(w / 2)
        new_pos[1] =  y + int(h / 2)

        curr_pos[0] += int(((((float(prev_pos[0]) + float(new_pos[0]))/2)) - curr_pos[0]) * damping)
        curr_pos[1] += int(((((float(prev_pos[1]) + float(new_pos[1]))/2)) - curr_pos[1]) * damping)

        cv2.circle(roi, (curr_pos[0] , curr_pos[1]), 10, color, 1)
        cv2.line(roi, (curr_pos[0], 0), (curr_pos[0], rows), color, 1)
        cv2.line(roi, (0, curr_pos[1]), (cols, curr_pos[1]), color, 1)

        trajectory.append([curr_pos[0], curr_pos[1]])

        if framenum > 1:
            traj_pos_x = int(abs(traj_offset_x - curr_pos[0])*1.5) + grid_x_pos
            traj_pos_y = int(abs(traj_offset_y - curr_pos[1])*2.25) + grid_y_pos

            trajectory_proj.append([traj_pos_x, traj_pos_y])

            #print(abs(traj_offset_y - curr_pos[1]))
            #print(np.exp(abs(traj_offset_y - curr_pos[1])))
            #print(prev_pos[0], curr_pos[1], ':', trajectory[framenum][0], trajectory[framenum - 1][0])

        break

    scale = 1.0

    draw_mobile_grid(roi, int(58*scale), int(104*scale), 1.0, (grid_x_pos, grid_y_pos), 7, 4, True, color_traj, 1)
    draw_trajectory(roi, trajectory, color_traj, 1, 1)

    if framenum > 1:
        draw_trajectory(roi, trajectory_proj, color_traj, 2, -1)
        draw_info(roi, trajectory, trajectory_proj, framenum, color)

    #cv2.imshow("Threshold", threshold)
    #cv2.imshow("gray roi", gray_roi)
    #print(size(threshold))
    combined = cv2.hconcat([roi, cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)])

    cv2.imshow("Roi", combined)
    video_out_tracked_dlib.write(roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()