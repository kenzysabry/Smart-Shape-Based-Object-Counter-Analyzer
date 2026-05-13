"""Simple centroid-based multi-object tracker with stable class voting."""

from dataclasses import dataclass, field
from math import hypot

import numpy as np

import config


@dataclass
class TrackedObject:
    """State stored for each tracked object."""

    object_id: int
    class_name: str
    box: tuple
    confidence: float
    centroid: tuple
    previous_centroid: tuple = None
    speed: float = 0.0
    missing_frames: int = 0
    class_scores: dict = field(default_factory=dict)


class CentroidTracker:
    """Tracks detections by matching their centroids across frames."""

    def __init__(
        self,
        max_distance=config.MAX_DISTANCE,
        max_missing_frames=config.MAX_MISSING_FRAMES,
    ):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.next_object_id = 1
        self.objects = {}

    @staticmethod
    def calculate_centroid(box):
        """Return the center point of a bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def calculate_speed(previous_centroid, current_centroid):
        """Estimate speed as centroid displacement in pixels per frame."""
        if previous_centroid is None:
            return 0.0
        return hypot(
            current_centroid[0] - previous_centroid[0],
            current_centroid[1] - previous_centroid[1],
        )

    @staticmethod
    def calculate_iou(first_box, second_box):
        """Return intersection-over-union for two bounding boxes."""
        first_x1, first_y1, first_x2, first_y2 = first_box
        second_x1, second_y1, second_x2, second_y2 = second_box

        inter_x1 = max(first_x1, second_x1)
        inter_y1 = max(first_y1, second_y1)
        inter_x2 = min(first_x2, second_x2)
        inter_y2 = min(first_y2, second_y2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
        second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def update_class_vote(tracked_object, class_name, confidence):
        """Smooth class labels over time so one weak frame cannot flip a track."""
        for existing_class in list(tracked_object.class_scores):
            tracked_object.class_scores[existing_class] *= config.CLASS_SCORE_DECAY

        tracked_object.class_scores[class_name] = (
            tracked_object.class_scores.get(class_name, 0.0) + confidence
        )
        tracked_object.class_name = max(
            tracked_object.class_scores,
            key=tracked_object.class_scores.get,
        )
        tracked_object.confidence = confidence

    def register(self, detection):
        """Create a new tracked object with a stable unique ID."""
        centroid = self.calculate_centroid(detection["box"])
        tracked_object = TrackedObject(
            object_id=self.next_object_id,
            class_name=detection["class_name"],
            box=detection["box"],
            confidence=detection["confidence"],
            centroid=centroid,
        )
        tracked_object.class_scores[detection["class_name"]] = detection["confidence"]
        self.objects[self.next_object_id] = tracked_object
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object that disappeared for too many frames."""
        self.objects.pop(object_id, None)

    def update(self, detections):
        """Match detections to existing objects and return tracked objects."""
        if not detections:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].missing_frames += 1
                if self.objects[object_id].missing_frames > self.max_missing_frames:
                    self.deregister(object_id)
            return self.get_objects()

        if not self.objects:
            for detection in detections:
                self.register(detection)
            return self.get_objects()

        object_ids = list(self.objects.keys())
        object_centroids = np.array(
            [self.objects[object_id].centroid for object_id in object_ids],
            dtype=np.float32,
        )
        detection_centroids = np.array(
            [self.calculate_centroid(detection["box"]) for detection in detections],
            dtype=np.float32,
        )

        pairs = []
        for object_index, object_centroid in enumerate(object_centroids):
            tracked_object = self.objects[object_ids[object_index]]
            for detection_index, detection_centroid in enumerate(detection_centroids):
                distance = np.linalg.norm(object_centroid - detection_centroid)
                iou = self.calculate_iou(
                    tracked_object.box,
                    detections[detection_index]["box"],
                )
                class_penalty = (
                    0.0
                    if detections[detection_index]["class_name"] == tracked_object.class_name
                    else 0.35
                )
                match_score = (distance / self.max_distance) + (1.0 - iou) + class_penalty
                pairs.append((match_score, distance, iou, object_index, detection_index))

        used_objects = set()
        used_detections = set()

        for _, distance, iou, object_index, detection_index in sorted(
            pairs,
            key=lambda item: item[0],
        ):
            if distance > self.max_distance and iou < 0.05:
                continue
            if object_index in used_objects or detection_index in used_detections:
                continue

            object_id = object_ids[object_index]
            detection = detections[detection_index]
            current_centroid = self.calculate_centroid(detection["box"])
            tracked_object = self.objects[object_id]

            tracked_object.previous_centroid = tracked_object.centroid
            tracked_object.centroid = current_centroid
            tracked_object.speed = self.calculate_speed(
                tracked_object.previous_centroid,
                current_centroid,
            )
            tracked_object.box = detection["box"]
            self.update_class_vote(
                tracked_object,
                detection["class_name"],
                detection["confidence"],
            )
            tracked_object.missing_frames = 0

            used_objects.add(object_index)
            used_detections.add(detection_index)

        for object_index, object_id in enumerate(object_ids):
            if object_index in used_objects:
                continue
            self.objects[object_id].missing_frames += 1
            if self.objects[object_id].missing_frames > self.max_missing_frames:
                self.deregister(object_id)

        for detection_index, detection in enumerate(detections):
            if detection_index not in used_detections:
                self.register(detection)

        return self.get_objects()

    def get_objects(self):
        """Return tracked objects sorted by their IDs."""
        return [self.objects[object_id] for object_id in sorted(self.objects.keys())]
