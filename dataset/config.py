import numpy as np

camera = "realsense"

ORI_WIDTH = 1280
ORI_HEIGHT = 720
width = ORI_WIDTH
height = ORI_HEIGHT

def set_resolution(w, h):
    global width, height
    width = w
    height = h

def get_camera_intrinsic():
    intrinsics = np.array([ # resize intrinsics: https://answers.opencv.org/question/150551
        (np.array([927.16973877, 0           , 651.31506348]) * (width / ORI_WIDTH)),
        (np.array([0           , 927.36688232, 349.62133789]) * (height / ORI_HEIGHT)),
        (np.array([0           , 0           , 1           ]))
    ])
    return intrinsics
