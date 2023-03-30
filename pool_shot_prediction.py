import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from intersectionandcheck import *
# Initializing Variables
timeout = 0
check = False
ballin = False
initialize=0

# Function to draw a dotted line
def dottedLine(img,pt1,pt2,color):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,15):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)
    for p in pts:
            cv2.circle(img,p,3,color,-1)
cap = cv2.VideoCapture('Shot-Predictor-Video.mp4')

if (cap.isOpened()==False):
    print("Error Opening the Video Stream or File")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('Pool_Shot_Prediction.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))


while (True):
    ret, frame = cap.read()
    if ret==True:
        if timeout > 0:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            timeout -= 1
            out.write(frame)
            continue
        # As cv2 reads the image in the BGR format, in the first step we will convert the image from BGR to HSV Color Space
        HSVSPACE = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Thresholding the frame to get only boundary walls of the pool
        mask_green = cv2.inRange(HSVSPACE, np.array([56, 161, 38]), np.array([71, 255, 94]))

        # Perform binary close and binary open operations using the kernels
        imageopen = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        imageclose = cv2.morphologyEx(imageopen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)))

        lines = cv2.HoughLinesP(imageclose, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        # Detect horizontal and vertical lines by checking their angles
        angle_threshold = 1
        horizontal_lines = []
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < angle_threshold:
                horizontal_lines.append(line)
            elif angle > 90 - angle_threshold and angle < 90 + angle_threshold:
                vertical_lines.append(line)

        middle_y = frame.shape[0] // 2
        top_lines = [line for line in horizontal_lines if line[0][1] < middle_y]
        bot_lines = [line for line in horizontal_lines if line[0][1] > middle_y]
        if len(bot_lines) == 0 or len(top_lines) == 0:
            out.write(frame)
            continue
        top_line = sorted(top_lines, key=lambda x: x[0][1])[0]
        bot_line = sorted(bot_lines, key=lambda x: x[0][1])[-1]

        middle_x = frame.shape[1] // 2
        left_lines = [line for line in vertical_lines if line[0][0] < middle_x]
        right_lines = [line for line in vertical_lines if line[0][0] > middle_x]
        if len(left_lines) == 0 or len(right_lines) == 0:
            out.write(frame)
            continue
        left_line = sorted(left_lines, key=lambda x: x[0][1])[0]
        right_line = sorted(right_lines, key=lambda x: x[0][1])[-1]

        top_left = [left_line[0][0], top_line[0][1]]
        bot_left = [left_line[0][0], bot_line[0][1]]
        top_right = [right_line[0][0], top_line[0][1]]
        bot_right = [right_line[0][0], bot_line[0][1]]

        corners = [top_left, bot_left, bot_right, top_right]
        w_rect = [[top_left[0] + 15, top_left[1] + 70], [bot_left[0] + 15, bot_left[1] - 70],[bot_right[0] - 15, bot_right[1] - 70], [top_right[0] - 15, top_right[1] + 70]]
        h_rect = [[top_left[0] + 70, top_left[1] + 15], [bot_left[0] + 70, bot_left[1] - 15],[bot_right[0] - 70, bot_right[1] - 15], [top_right[0] - 70, top_right[1] + 15]]

        mid = int((top_left[0] + top_right[0]) / 2)

        pockets = [top_left, top_right, bot_left, bot_right, [mid, top_left[1]], [mid, bot_left[1]]]
        pockets = np.array(pockets)

        top_p_rect = [[mid - 60, top_left[1]], [mid - 60, top_left[1] + 60],
                      [mid + 60, top_left[1] + 60], [mid + 60, top_left[1]]]
        bot_p_rect = [[mid - 60, bot_left[1]], [mid - 60, bot_left[1] - 60],
                      [mid + 60, bot_left[1] - 60], [mid + 60, bot_left[1]]]

        # Threshold the image to get only white pixels
        mask = np.zeros(HSVSPACE.shape[:2], dtype=np.uint8)

        # Create a filled polygon with the four corners to define the region of interest
        w_corners = np.array([w_rect], dtype=np.int32)
        h_corners = np.array([h_rect], dtype=np.int32)
        cv2.fillPoly(mask, w_corners, 255)
        cv2.fillPoly(mask, h_corners, 255)
        top_p_rect = np.array([top_p_rect], dtype=np.int32)
        bot_p_rect = np.array([bot_p_rect], dtype=np.int32)
        cv2.fillPoly(mask, top_p_rect, 0)
        cv2.fillPoly(mask, bot_p_rect, 0)

        # Apply the mask to the thresholding operation
        board_mask = cv2.bitwise_and(HSVSPACE, HSVSPACE, mask=mask)

        white_mask = cv2.inRange(board_mask, np.array([10, 14, 144]), np.array([100, 42, 255]))
        board_mask = cv2.inRange(board_mask, np.array([56, 131, 4]), np.array([75, 221, 215]))
        mask_inv = cv2.bitwise_not(board_mask, mask=mask)
        lines = cv2.HoughLinesP(white_mask, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=100,
                                maxLineGap=50)
        best_line = None
        if lines is not None:
            best_len = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > best_len:
                    best_len = length
                    best_line = (x1, y1, x2, y2)
        contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = 0
        circ_pos = []
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 22 and radius < 35:
                cv2.circle(frame, center, radius, (222,82,175), -1)
                circles += 1
                circ_pos.append([np.array(center), radius])

        if circles < 2:
            gray = cv2.bitwise_and(frame, frame, mask=mask)
            ball = cv2.imread('pic1.png')
            w_ball = cv2.imread('pic3.png')
            for pattern in [ball, w_ball]:
                w, h = pattern.shape[:-1]
                res = cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED)
                threshold = 0.4
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
                    circles += 1
                    cv2.circle(frame, center, 25, (222,82,175), -1)
                    circ_pos.append([np.array(center), 25])
                    break

        if circles == 2 and best_line is not None:
            # Calculate resulting direction of circle furthest away from line
            avg_cue = np.array([(best_line[2] + best_line[0]) / 2, (best_line[3] + best_line[1]) / 2])
            if np.linalg.norm(circ_pos[0][0] - avg_cue) < np.linalg.norm(circ_pos[1][0] - avg_cue):
                ball = circ_pos[0]
                target = circ_pos[1]
            else:
                ball = circ_pos[1]
                target = circ_pos[0]
            if np.linalg.norm(np.array([9999, 9999]) - ball[0]) > 5 and np.linalg.norm(np.array([9999, 9999]) - ball[0]) < 100:
                print("Go ahead with predictions")
            checking=np.array([9999, 9999])
            checking = ball[0]
            cue_start = np.array([best_line[0], best_line[1]])
            cue_end = np.array([best_line[2], best_line[3]])
            if np.linalg.norm(cue_start - ball[0]) < np.linalg.norm(cue_end - ball[0]):
                temp = cue_end
                cue_end = cue_start
                cue_start = temp

            cue_dir = (cue_end - cue_start) / np.linalg.norm(cue_end - cue_start)
            v = ball[0] - cue_end
            proj = np.dot(v, cue_dir) * cue_dir
            ball = [cue_end + proj, ball[1]]

            dist_along_cue = np.dot(target[0] - ball[0], cue_dir)
            if dist_along_cue < 0:
                collision_point = ball[0] + (dist_along_cue + ball[1] + target[1]) * cue_dir
            else:
                collision_point = ball[0] + (dist_along_cue - ball[1] - target[1]) * cue_dir
            cv2.line(frame, (int(ball[0][0]), int(ball[0][1])), (int(collision_point[0]), int(collision_point[1])),
                     (222,82,175), thickness=2)

            movement_dir = target[0] - collision_point
            intersection, normal = findintersection(movement_dir, top_left, bot_right, target[0])
            hit = insidepocket(pockets, intersection)
            if not hit:
                # See if the ball target ball can reflect off a surface
                perp = np.dot(movement_dir, normal) * normal
                reflection = movement_dir - 2 * perp
                new_dir = reflection / np.linalg.norm(reflection)
                old_intersection = intersection
                intersection, normal = findintersection(new_dir, top_left, bot_right, intersection - movement_dir * 0.1)
                if insidepocket(pockets, intersection):
                    cv2.circle(frame, intersection, 25, (222,82,175), -1)
                    ballin = True
                cv2.line(frame, (int(collision_point[0]), int(collision_point[1])),
                         (int(old_intersection[0]), int(old_intersection[1])), (222,82,175), thickness=2)
                cv2.line(frame, (int(old_intersection[0]), int(old_intersection[1])),
                         (int(intersection[0]), int(intersection[1])), (222,82,175), thickness=2)
                #dottedLine(frame, (int(collision_point[0]), int(collision_point[1])),
                 #        (int(old_intersection[0]), int(old_intersection[1])), (222,82,175))
                #dottedLine(frame, (int(old_intersection[0]), int(old_intersection[1])),
                #         (int(intersection[0]), int(intersection[1])), (222,82,175))
            else:
                cv2.circle(frame, intersection, 25, (222,82,175), -1)
                ballin = True
                cv2.line(frame, (int(collision_point[0]), int(collision_point[1])),
                         (int(intersection[0]), int(intersection[1])), (222,82,175), thickness=2)
                #dottedLine(frame, (int(collision_point[0]), int(collision_point[1])),
                #         (int(intersection[0]), int(intersection[1])), (222,82,175))
            if not ballin:
                cv2.circle(frame, intersection, 25, (222,82,175), -1)
                fontpath = "sfpro.ttf"
                font = ImageFont.truetype(fontpath, 40)
                im = Image.fromarray(frame)
                draw = ImageDraw.Draw(im)
                draw.rounded_rectangle((89, 26, 430, 113), fill=(84, 61, 246), radius=40)
                draw.text((113, 45), "Prediction: OUT", font=font, fill=(255, 255, 255))
                frame = np.array(im)
            else:
                fontpath = "sfpro.ttf"
                font1 = ImageFont.truetype(fontpath, 40)
                im = Image.fromarray(frame)
                draw = ImageDraw.Draw(im)
                draw.rounded_rectangle((89, 26, 380, 113), fill=(254, 118, 136), radius=40)
                draw.text((112, 46), "Prediction: IN",  font=font1, fill=(255, 255, 255))
                frame = np.array(im)
            timelimits = [95, 130, 123, 150, 74, 175, 120, 135, 102, 100]
            timeout = timelimits[initialize]
            initialize += 1
            check = True

        # Show the final image with all contours
        cv2.imshow("frame", frame)
        if check:
            for i in range(20):
                out.write(frame)
            check = False
            ballin = False
        # Wait for user to close the window
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
