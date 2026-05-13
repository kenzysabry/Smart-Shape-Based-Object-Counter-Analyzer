"""Contour-based object detection and rule-based classification."""

import cv2

import config


class ContourObjectDetector:
    """Detect moving objects from a motion mask and classify them by rules."""

    def __init__(
        self,
        min_contour_area=config.MIN_CONTOUR_AREA,
        min_box_width=config.MIN_BOX_WIDTH,
        min_box_height=config.MIN_BOX_HEIGHT,
        merge_margin=config.BOX_MERGE_MARGIN,
    ):
        self.min_contour_area = min_contour_area
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.merge_margin = merge_margin

    def detect(self, frame, motion_mask):
        """Return moving-object detections based on contours."""
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_width * frame_height)
        detections = []

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contour_boxes = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < self.min_contour_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < self.min_box_width or height < self.min_box_height:
                continue

            contour_boxes.append((x, y, x + width, y + height, contour_area))

        merged_boxes = self._merge_boxes(contour_boxes, frame_width, frame_height)

        for x1, y1, x2, y2, contour_area in merged_boxes:
            width = x2 - x1
            height = y2 - y1
            features = self._extract_features(
                contour_area,
                width,
                height,
                frame_width,
                frame_height,
                frame_area,
            )
            class_name, confidence = self.classify(features)

            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "raw_class_name": "motion_contour",
                    "class_id": 0,
                    "confidence": confidence,
                    "features": features,
                }
            )

        return detections

    def _merge_boxes(self, boxes, frame_width, frame_height):
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            result = []
            used = [False] * len(merged)

            for index, box in enumerate(merged):
                if used[index]:
                    continue

                current = box
                used[index] = True

                for other_index in range(index + 1, len(merged)):
                    if used[other_index]:
                        continue
                    other = merged[other_index]
                    if not self._boxes_touch(current, other):
                        continue

                    current = self._union_boxes(current, other, frame_width, frame_height)
                    used[other_index] = True
                    changed = True

                result.append(current)

            merged = result

        return merged

    def _boxes_touch(self, first, second):
        fx1, fy1, fx2, fy2, _ = first
        sx1, sy1, sx2, sy2, _ = second
        margin = self.merge_margin
        return not (
            fx2 + margin < sx1
            or sx2 + margin < fx1
            or fy2 + margin < sy1
            or sy2 + margin < fy1
        )

    @staticmethod
    def _union_boxes(first, second, frame_width, frame_height):
        fx1, fy1, fx2, fy2, first_area = first
        sx1, sy1, sx2, sy2, second_area = second
        return (
            max(0, min(fx1, sx1)),
            max(0, min(fy1, sy1)),
            min(frame_width, max(fx2, sx2)),
            min(frame_height, max(fy2, sy2)),
            first_area + second_area,
        )

    def refine_tracked_object(self, tracked_object, shape_info, frame_shape):
        """Update a tracked object's label using size, shape, and movement."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = tracked_object.box
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        box_area = float(box_width * box_height)
        frame_area = float(width * height)

        features = {
            "area_ratio": box_area / frame_area,
            "aspect_ratio": box_width / float(box_height),
            "width_ratio": box_width / float(width),
            "height_ratio": box_height / float(height),
            "fill_ratio": 1.0,
            "speed": tracked_object.speed,
            "line_count": shape_info["lines"],
            "circle_count": shape_info["circles"],
        }

        class_name, confidence = self.classify(features)
        tracked_object.class_name = class_name
        tracked_object.confidence = confidence

    @staticmethod
    def _extract_features(contour_area, width, height, frame_width, frame_height, frame_area):
        box_area = float(width * height)
        return {
            "area_ratio": box_area / frame_area,
            "aspect_ratio": width / float(max(1, height)),
            "width_ratio": width / float(frame_width),
            "height_ratio": height / float(frame_height),
            "fill_ratio": contour_area / box_area if box_area else 0.0,
            "speed": 0.0,
            "line_count": 0,
            "circle_count": 0,
        }

    @staticmethod
    def classify(features):
        """Classify by size, shape, and movement rules."""
        area_ratio = features["area_ratio"]
        aspect_ratio = features["aspect_ratio"]
        width_ratio = features["width_ratio"]
        height_ratio = features["height_ratio"]
        fill_ratio = features["fill_ratio"]
        speed = features["speed"]
        line_count = features["line_count"]
        circle_count = features["circle_count"]

        strong_person_shape = aspect_ratio <= config.PERSON_STRICT_ASPECT_RATIO
        possible_person_shape = aspect_ratio <= config.PERSON_MAX_ASPECT_RATIO
        person_size = area_ratio <= config.PERSON_LARGE_AREA_RATIO
        person_motion = speed >= config.FAST_OBJECT_SPEED

        if person_size and (strong_person_shape or (possible_person_shape and person_motion)):
            confidence = min(
                0.92,
                0.55 + speed * 0.02 + max(0.0, 1.0 - aspect_ratio) * 0.25,
            )
            return "Person", confidence

        car_size = (
            area_ratio >= config.CAR_MIN_AREA_RATIO
            and width_ratio >= config.CAR_MIN_WIDTH_RATIO
            and height_ratio >= config.CAR_MIN_HEIGHT_RATIO
        )
        car_shape = aspect_ratio >= config.VEHICLE_ASPECT_RATIO
        hough_vehicle_detail = (
            circle_count >= config.VEHICLE_MIN_CIRCLES
            or line_count >= config.VEHICLE_MIN_LINES
        )
        large_vehicle_like = (
            area_ratio >= config.LARGE_OBJECT_AREA_RATIO
            and aspect_ratio >= config.PERSON_MAX_ASPECT_RATIO
        )

        moving_wide_object = car_shape and speed >= config.CAR_MIN_SPEED_FOR_SHAPE_ONLY
        shaped_vehicle = car_shape and hough_vehicle_detail

        if car_size and (
            moving_wide_object
            or shaped_vehicle
            or (large_vehicle_like and hough_vehicle_detail)
        ):
            confidence = min(
                0.95,
                0.55 + area_ratio * 12 + max(0.0, aspect_ratio - 1.0) * 0.12,
            )
            return "Car", confidence

        return "Moving Object", 0.50
