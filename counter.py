"""Virtual line crossing counter."""

import cv2

import config
from motion import MotionDetector


class ObjectCounter:
    """Counts IN/OUT crossings and currently moving objects."""

    def __init__(self, line_y=config.COUNTING_LINE_Y):
        self.line_y = line_y
        self.total_in = 0
        self.total_out = 0
        self.class_counts = {class_name: 0 for class_name in config.COUNTED_CLASSES}
        self.moving_class_counts = {
            class_name: 0 for class_name in config.COUNTED_CLASSES
        }
        self.moving_total = 0
        self.moving_vehicle_total = 0
        self.counted_ids = set()

    def update(self, tracked_objects):
        """Update counters when objects cross the horizontal virtual line."""
        for tracked_object in tracked_objects:
            if tracked_object.missing_frames > 0:
                continue
            if tracked_object.previous_centroid is None:
                continue
            if tracked_object.object_id in self.counted_ids:
                continue

            previous_y = tracked_object.previous_centroid[1]
            current_y = tracked_object.centroid[1]

            if previous_y < self.line_y <= current_y:
                self.total_in += 1
                self._increment_class_count(tracked_object.class_name)
                self.counted_ids.add(tracked_object.object_id)

            elif previous_y > self.line_y >= current_y:
                self.total_out += 1
                self._increment_class_count(tracked_object.class_name)
                self.counted_ids.add(tracked_object.object_id)

    def update_moving_counts(self, tracked_objects, motion_mask):
        """Count visible tracked objects that are moving in the current frame."""
        self.moving_total = 0
        self.moving_vehicle_total = 0
        for class_name in self.moving_class_counts:
            self.moving_class_counts[class_name] = 0

        for tracked_object in tracked_objects:
            if tracked_object.missing_frames > 0:
                continue

            motion_ratio = MotionDetector.motion_ratio_in_box(
                motion_mask,
                tracked_object.box,
            )
            if motion_ratio < config.MOTION_RATIO_THRESHOLD:
                continue

            self.moving_total += 1
            if tracked_object.class_name in self.moving_class_counts:
                self.moving_class_counts[tracked_object.class_name] += 1
            if tracked_object.class_name in config.VEHICLE_CLASSES:
                self.moving_vehicle_total += 1

    def _increment_class_count(self, class_name):
        if class_name in self.class_counts:
            self.class_counts[class_name] += 1

    def draw_line(self, frame):
        """Draw the horizontal virtual counting line."""
        height, width = frame.shape[:2]
        visible_line_y = max(0, min(height - 1, self.line_y))

        cv2.line(
            frame,
            (0, visible_line_y),
            (width, visible_line_y),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "Counting Line",
            (10, max(25, visible_line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def draw_counts(self, frame):
        """Draw total and class counters on the output frame."""
        lines = [
            f"IN: {self.total_in}",
            f"OUT: {self.total_out}",
            f"Moving: {self.moving_total}",
            f"Moving Vehicles: {self.moving_vehicle_total}",
            f"Moving Cars: {self.moving_class_counts['Car']}",
        ]
        lines.extend(
            f"{class_name}: {self.class_counts[class_name]}"
            for class_name in config.COUNTED_CLASSES
        )

        x, y = 15, 30
        line_height = 28
        panel_width = 290
        panel_height = 20 + line_height * len(lines)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - 25),
            (x - 10 + panel_width, y - 25 + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
