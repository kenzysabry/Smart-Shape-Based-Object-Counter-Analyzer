"""Canny edge detection and Hough Transform analysis for object ROIs."""

import cv2
import numpy as np

import config


class ShapeAnalyzer:
    """Counts Hough lines and circles inside each tracked object ROI."""

    def __init__(self, canny_low=config.CANNY_LOW, canny_high=config.CANNY_HIGH):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def analyze(self, frame, box, draw=True):
        """Analyze one object ROI and optionally draw Hough results."""
        x1, y1, x2, y2 = self._clip_box(box, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return {"lines": 0, "circles": 0}

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {"lines": 0, "circles": 0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        edges = self._detect_edges(blurred)
        roi_width = x2 - x1
        roi_height = y2 - y1

        lines = self._detect_lines(edges, roi_width, roi_height)
        circles = self._detect_circles(blurred, edges, roi_width, roi_height)

        line_count = len(lines)
        circle_count = len(circles)

        if draw:
            self._draw_lines(frame, lines, x1, y1)
            self._draw_circles(frame, circles, x1, y1)

        return {
            "lines": int(line_count),
            "circles": int(circle_count),
        }

    def _detect_edges(self, blurred_gray):
        median_intensity = float(np.median(blurred_gray))
        if median_intensity <= 0:
            return cv2.Canny(blurred_gray, self.canny_low, self.canny_high)

        lower = int(max(10, (1.0 - config.CANNY_AUTO_SIGMA) * median_intensity))
        upper = int(min(255, (1.0 + config.CANNY_AUTO_SIGMA) * median_intensity))
        if upper <= lower:
            lower, upper = self.canny_low, self.canny_high

        return cv2.Canny(blurred_gray, lower, upper)

    def _detect_lines(self, edges, roi_width, roi_height):
        max_dimension = max(roi_width, roi_height)
        min_line_length = max(
            18,
            int(max_dimension * config.HOUGH_MIN_LINE_LENGTH_RATIO),
        )
        max_line_gap = max(5, int(max_dimension * config.HOUGH_MAX_LINE_GAP_RATIO))
        threshold = max(25, int(min_line_length * 0.85))

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if raw_lines is None:
            return []

        filtered_lines = []
        seen_buckets = set()
        for line in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_line_length:
                continue

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bucket = (
                int(round(angle / 12.0)),
                center_x // 24,
                center_y // 24,
            )
            if bucket in seen_buckets:
                continue

            seen_buckets.add(bucket)
            filtered_lines.append((length, x1, y1, x2, y2))

        filtered_lines.sort(reverse=True)
        return [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in filtered_lines]

    def _detect_circles(self, blurred_gray, edges, roi_width, roi_height):
        min_dimension = min(roi_width, roi_height)
        if min_dimension < 32:
            return []

        min_radius = max(5, int(min_dimension * 0.08))
        max_radius = max(min_radius + 2, min(45, int(min_dimension * 0.32)))
        circle_source = cv2.medianBlur(blurred_gray, 5)

        raw_circles = cv2.HoughCircles(
            circle_source,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=max(24, int(min_dimension * 0.32)),
            param1=70,
            param2=36,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if raw_circles is None:
            return []

        candidates = np.uint16(np.around(raw_circles[0]))
        filtered_circles = []
        for cx, cy, radius in candidates:
            circle = (int(cx), int(cy), int(radius))
            if not self._has_enough_edge_support(edges, circle):
                continue
            if self._is_duplicate_circle(filtered_circles, circle):
                continue
            filtered_circles.append(circle)

        return filtered_circles

    @staticmethod
    def _has_enough_edge_support(edges, circle):
        cx, cy, radius = circle
        height, width = edges.shape[:2]
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        supported_points = 0
        valid_points = 0

        for angle in angles:
            x = int(round(cx + radius * np.cos(angle)))
            y = int(round(cy + radius * np.sin(angle)))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            valid_points += 1
            neighborhood = edges[
                max(0, y - 1) : min(height, y + 2),
                max(0, x - 1) : min(width, x + 2),
            ]
            if cv2.countNonZero(neighborhood) > 0:
                supported_points += 1

        if valid_points == 0:
            return False

        support_ratio = supported_points / float(valid_points)
        return support_ratio >= config.CIRCLE_EDGE_SUPPORT_THRESHOLD

    @staticmethod
    def _is_duplicate_circle(existing_circles, circle):
        cx, cy, radius = circle
        for existing_cx, existing_cy, existing_radius in existing_circles:
            center_distance = np.hypot(cx - existing_cx, cy - existing_cy)
            radius_difference = abs(radius - existing_radius)
            if center_distance < max(8, radius * 0.45) and radius_difference < 6:
                return True
        return False

    @staticmethod
    def _clip_box(box, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height))
        return x1, y1, x2, y2

    @staticmethod
    def _draw_lines(frame, lines, offset_x, offset_y):
        if not lines:
            return

        for line in lines[: config.MAX_DRAWN_HOUGH_LINES]:
            x1, y1, x2, y2 = line
            cv2.line(
                frame,
                (offset_x + x1, offset_y + y1),
                (offset_x + x2, offset_y + y2),
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_circles(frame, circles, offset_x, offset_y):
        if not circles:
            return

        for circle in circles[: config.MAX_DRAWN_HOUGH_CIRCLES]:
            cx, cy, radius = circle
            center = (offset_x + int(cx), offset_y + int(cy))
            cv2.circle(frame, center, int(radius), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center, 2, (0, 255, 255), 2, cv2.LINE_AA)
