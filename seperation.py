"""
author: LAI JUNJIE

face landmarks:
    jaw;       // [0 , 16]
    rightBrow; // [17, 21]
    leftBrow;  // [22, 26]
    nose;      // [27, 35]
    rightEye;  // [36, 41]
    leftEye;   // [42, 47]
    mouth;     // [48, 59]
    mouth2;    // [60, 67]
"""
import cv2
import dlib
import os
import sys
from utilis.utilis import *

image_path = './images'
save_path = './result'

images = os.listdir(image_path)


for images in images:
    img = cv2.imread(os.path.join(image_path, images))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = landmark_np(img_gray)

    # 嘴部
    mouth_x1, mouth_y1, mouth_x2, mouth_y2 = find_position(landmarks, 48, 67)
    save_img(img, mouth_x1, mouth_y1, mouth_x2, mouth_y2, os.path.join(save_path, images[:-4] + '_mouth.jpg'))

    # 左眉毛
    leftbrow_x1, leftbrow_y1, leftbrow_x2, left_brow_y2 = find_position(landmarks, 22, 26)
    save_img(img, leftbrow_x1, leftbrow_y1, leftbrow_x2, left_brow_y2, os.path.join(save_path, images[:-4] + '_leftbrow.jpg'))

    # 右眉毛
    rightbrow_x1, rightbrow_y1, rightbrow_x2, rightbrow_y2 = find_position(landmarks, 17, 21)
    save_img(img, rightbrow_x1, rightbrow_y1, rightbrow_x2, rightbrow_y2, os.path.join(save_path, images[:-4] + '_rightbrow.jpg'))

    # 左眼
    lefteye_x1, lefteye_y1, lefteye_x2, lefteye_y2 = find_position(landmarks, 42, 47)
    save_img(img, lefteye_x1, lefteye_y1, lefteye_x2, lefteye_y2, os.path.join(save_path, images[:-4] + '_lefteye.jpg'))

    # 右眼
    righteye_x1, righteye_y1, righteye_x2, righteye_y2 = find_position(landmarks, 36, 41)
    save_img(img, righteye_x1, righteye_y1, righteye_x2, righteye_y2, os.path.join(save_path, images[:-4] + '_righteye.jpg'))
