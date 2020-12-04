from typing import List
import numpy as np
import cv2


class MaxOver:
    def __init__(self, number_frames: int):
        self._number_frames = number_frames

    def __call__(self, frame_set: List[np.ndarray]) -> np.ndarray:
        frame_subset = frame_set[-self._number_frames :]
        maximum = np.zeros_like(frame_set[0])
        for frame in frame_subset:
            maximum = np.maximum(maximum, frame)

        return maximum


class DownSample:
    def __init__(self, width: int, height: int):
        self._dimensions = (width, height)

    def __call__(self, original_image: np.ndarray):
        return cv2.resize(original_image, self._dimensions)


class GrayScale:
    def __call__(self, rgb_image: np.ndarray):
        return np.mean(rgb_image, axis=2).astype(np.uint8)