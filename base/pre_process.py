import cv2
import numpy as np

Mat = np.ndarray[int, np.dtype[np.generic]]


def pre_yolact(frame: Mat, net_size:int):
    """
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net_size = (net_size, net_size)
    frame = cv2.resize(frame,net_size)
    return frame