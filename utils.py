import numpy as np
import cv2

def inch_2_m(x):
    return np.array(x) * 0.0254


def m_2_inch(x):
    return np.array(x) * 39.37

def dehomogenize_3d(vec):
    vec = vec.reshape((-1,1))
    vec = vec/vec[3]
    return vec[:3]

def dehomogenize_2d(vec):
    vec = vec.reshape((-1,1))
    vec = vec/vec[2]
    return vec[:2]

