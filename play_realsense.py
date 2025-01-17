"""Game of tic tac toe using OpenCV to play against computer"""


import os
import sys
import cv2
import argparse
import numpy as np

from tensorflow.python.keras.models import load_model

from utils import imutils
from utils import detections
from alphabeta import Tic, get_enemy, determine
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Load model
# model = load_model("/home/smr/tic-tac-toe/tic-tac-toe/data/model.h5")

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


def find_board(frame):
    
        frame_cp = frame.copy()
        
        gray = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2GRAY)
        cv2.imshow('RealSense_gray', gray)
        
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        
        gray = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
        # Convert to float32
        gray = np.float32(gray)
        
        # Apply Harris corner detection
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        # Result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        
        # Threshold for an optimal value, it may vary depending on the image.
        frame_cp[dst > 0.05 * dst.max()] = [0, 0, 255]
        
        # Display the result
        cv2.imshow('Corners_all', frame_cp)
        
        return frame_cp

def adjust_threshold(thresh):
    # Implement your logic to adjust the threshold here
    # For example, you might want to increase or decrease the threshold value
    return thresh

def find_contour(frame):
    frame_cp = frame.copy()
    
        # Create a grayscale image
    gray = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    #cv2.imshow('gray', gray)
    # Apply adaptive thresholding
    gray_threshold = cv2.adaptiveThreshold(
        gray, 75, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5
    )
    # Display the result
    cv2.imshow('gray_threshold', gray_threshold)

        # Find contours
    contours, _ = cv2.findContours(
        gray_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the contour with the largest area
        max_contour = max(contours, key=cv2.contourArea)
        print("max_contour ", max_contour)
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # Check if the polygon has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Draw the corners on the image
            for point in approx:
                cv2.circle(frame_cp, tuple(point[0]), 10, (0, 255, 0), -1)

        # Display the result
        cv2.imshow('corners', frame_cp)

    return frame_cp



def play_realsense(frame):
    board = Tic()
    history = {}
    message = True
    cv2.imshow('RealSense', frame)
    cv2.waitKey(1)
    
    while True:
        print("in the loop play_realsense")
        
        a = find_contour(frame)
        #cv2.imshow('contour', a)
        
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def main_realsense():
    try:
        while True:
            
            #delay 1 

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            play_realsense(color_image)
            
            
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        sys.exit()
    


if __name__ == '__main__':
    main_realsense()