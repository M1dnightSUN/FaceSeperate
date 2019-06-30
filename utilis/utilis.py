# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import dlib

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./lib/shape_predictor_68_face_landmarks.dat')


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def landmark_np(img):
    faces = detector(img, 1)
    for i in range(len(faces)):
        # 取特征点坐标
        landmarks = np.array([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
        return landmarks


# 寻找某个部位的左上角和右下角
def find_position(landmarks, pos1, pos2):
    # 寻找嘴部的x, y, w, h
    x1 = min(landmarks[pos1:pos2, 0])
    y1 = min(landmarks[pos1:pos2, 1])
    x2 = max(landmarks[pos1:pos2, 0])
    y2 = max(landmarks[pos1:pos2, 1])

    return x1-10, y1-10, x2+10, y2+10


def save_img(img, x1, y1, x2, y2, savename):
    roi_img = img[y1:y2, x1:x2]
    cv2.imwrite(savename, roi_img)
    print('Save successfully!')


def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
