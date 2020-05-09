import cv2
import numpy as np


def process_frame(frame, shape=(84, 84)):
    """
    Using full-scale, RBG images for our network training
    is just too computationally demanding. We can reduce
    them to 84x84 grayscale images.
    Args:
        frame: The frame to process. Must have values ranging from 0-255.
        shape: The desired output shape.
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame
