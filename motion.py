"""Motion detection using grayscale conversion and background subtraction."""

import cv2
import numpy as np


class MotionDetector:
    """Extract moving regions with MOG2 and morphology."""

    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        """Return a clean binary motion mask for the current frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        foreground_mask = self.background_subtractor.apply(gray)

        _, threshold_mask = cv2.threshold(
            foreground_mask,
            200,
            255,
            cv2.THRESH_BINARY,
        )

        opened = cv2.morphologyEx(
            threshold_mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )

        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=1,
        )

        clean_mask = cv2.dilate(closed, self.kernel, iterations=1)
        return clean_mask

    @staticmethod
    def draw_motion_regions(frame, motion_mask, min_area=500):
        """Draw contours around moving areas for optional visualization."""
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

        return frame

    @staticmethod
    def motion_ratio_in_box(motion_mask, box):
        """Return the fraction of foreground pixels inside a bounding box."""
        x1, y1, x2, y2 = box
        height, width = motion_mask.shape[:2]
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = motion_mask[y1:y2, x1:x2]
        return cv2.countNonZero(roi) / float(roi.size)
