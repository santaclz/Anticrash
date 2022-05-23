import carla
import random
import queue
import numpy as np
import cv2
import time
#import torch
import matplotlib.pyplot as plt


### Lane detection
def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    return np.array([left_line, right_line])

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            except Exception:
                print("Drawing line failed")
    return line_image

def draw_lane(image):
    # Detect sharp edges
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # Get region of interest
    height = image.shape[0]
    polygons = np.array([[(100, height), (700, height), (400, 310)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(canny, mask[:,:,0])
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    avg_lines = avg_slope_intercept(lane_image, lines)
    drawn_lines = draw_lines(image, avg_lines)
    result = cv2.bitwise_or(drawn_lines, image)
    #result = cv2.addWeighted(draw_lines(image, lines), 1, lane_image, 1, 1) 
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #plt.imshow(canny)
    #plt.show()
    return result


