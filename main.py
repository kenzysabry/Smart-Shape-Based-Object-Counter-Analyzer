"""Main entry point for the Motion-Aware Smart Surveillance System."""

from pathlib import Path

import cv2

import config
from classification import ContourObjectDetector
from counter import ObjectCounter
from motion import MotionDetector
from shape_analysis import ShapeAnalyzer
from tracker import CentroidTracker


CLASS_COLORS = {
    "Person": (255, 0, 255),
    "Car": (0, 255, 0),
    "Moving Object": (0, 255, 255),
}


def resolve_project_path(relative_path):
    """Resolve config paths relative to this project folder."""
    return Path(__file__).resolve().parent / relative_path


def draw_tracked_object(frame, tracked_object, shape_info):
    """Draw bounding box, ID, class, confidence, speed, and Hough counts."""
    x1, y1, x2, y2 = tracked_object.box
    color = CLASS_COLORS.get(tracked_object.class_name, (255, 255, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, tracked_object.centroid, 4, color, -1)

    label_lines = [
        (
            f"ID {tracked_object.object_id} | "
            f"{tracked_object.class_name} | "
            f"{tracked_object.confidence:.2f}"
        ),
        f"Lines: {shape_info['lines']} | Circles: {shape_info['circles']}",
        f"Speed: {tracked_object.speed:.1f} px/frame",
    ]

    draw_label_block(frame, label_lines, (x1, y1), color)


def should_display_tracked_object(tracked_object, motion_mask):
    """Return True when the tracked object is visibly moving."""
    x1, y1, x2, y2 = tracked_object.box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    aspect_ratio = box_width / float(box_height)
    motion_ratio = MotionDetector.motion_ratio_in_box(motion_mask, tracked_object.box)

    if (
        aspect_ratio >= config.STATIC_WIDE_OBJECT_MIN_ASPECT
        and tracked_object.speed < config.STATIC_WIDE_OBJECT_MAX_SPEED
    ):
        return False

    if (
        aspect_ratio >= config.STATIC_WIDE_OBJECT_MIN_ASPECT
        and overlaps_static_vehicle_region(tracked_object.box)
    ):
        return False

    return (
        motion_ratio >= config.DISPLAY_MIN_MOTION_RATIO
        or tracked_object.speed >= config.DISPLAY_MIN_SPEED
    )


def overlaps_static_vehicle_region(box):
    """Return True when a wide object mostly belongs to a known parked zone."""
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))

    for region in config.STATIC_VEHICLE_IGNORE_REGIONS:
        rx1, ry1, rx2, ry2 = region
        overlap_x1 = max(x1, rx1)
        overlap_y1 = max(y1, ry1)
        overlap_x2 = min(x2, rx2)
        overlap_y2 = min(y2, ry2)
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            continue

        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        if overlap_area / float(box_area) >= config.STATIC_REGION_OVERLAP_THRESHOLD:
            return True

    return False


def draw_label_block(frame, label_lines, anchor, color):
    """Draw a readable text block above or inside the object box."""
    x, y = anchor
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 5
    line_gap = 7

    text_sizes = [
        cv2.getTextSize(text, font, font_scale, thickness)[0]
        for text in label_lines
    ]
    block_width = max(width for width, _ in text_sizes) + padding * 2
    block_height = sum(height for _, height in text_sizes)
    block_height += padding * 2 + line_gap * (len(label_lines) - 1)

    x = max(0, min(x, frame_width - block_width - 1))
    top_y = y - block_height - 4
    if top_y < 0:
        top_y = min(frame_height - block_height - 1, y + 4)

    cv2.rectangle(
        frame,
        (x, top_y),
        (x + block_width, top_y + block_height),
        color,
        -1,
    )

    cursor_y = top_y + padding
    for text, (_, text_height) in zip(label_lines, text_sizes):
        cursor_y += text_height
        cv2.putText(
            frame,
            text,
            (x + padding, cursor_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_gap


def create_video_writer(output_path, fps, width, height):
    """Create an MP4 writer for the processed output video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer: {output_path}")
    return writer


def process_video():
    """Run the full detection, tracking, counting, and output pipeline."""
    video_path = resolve_project_path(config.VIDEO_PATH)
    output_path = resolve_project_path(config.OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file was not found: {video_path}\n"
            "Place your input video at videos/video_project.mp4."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if config.SAVE_OUTPUT:
        writer = create_video_writer(output_path, fps, width, height)

    motion_detector = MotionDetector()
    object_detector = ContourObjectDetector()
    tracker = CentroidTracker()
    counter = ObjectCounter(line_y=height // 2)
    shape_analyzer = ShapeAnalyzer()

    display_enabled = True
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1

            motion_mask = motion_detector.detect(frame)

            detections = object_detector.detect(frame, motion_mask)
            tracked_objects = tracker.update(detections)

            output_frame = frame.copy()

            if config.SHOW_MOTION_MASK:
                motion_detector.draw_motion_regions(
                    output_frame,
                    motion_mask,
                    min_area=config.MIN_CONTOUR_AREA,
                )

            shape_info_by_id = {}
            visible_tracked_objects = []
            for tracked_object in tracked_objects:
                if tracked_object.missing_frames > 0:
                    continue

                shape_info = shape_analyzer.analyze(
                    frame,
                    tracked_object.box,
                    draw=False,
                )
                object_detector.refine_tracked_object(
                    tracked_object,
                    shape_info,
                    output_frame.shape,
                )

                if not should_display_tracked_object(tracked_object, motion_mask):
                    continue

                shape_info = shape_analyzer.analyze(
                    output_frame,
                    tracked_object.box,
                    draw=True,
                )
                shape_info_by_id[tracked_object.object_id] = shape_info
                visible_tracked_objects.append(tracked_object)

            counter.update(visible_tracked_objects)
            counter.update_moving_counts(visible_tracked_objects, motion_mask)

            for tracked_object in visible_tracked_objects:
                draw_tracked_object(
                    output_frame,
                    tracked_object,
                    shape_info_by_id[tracked_object.object_id],
                )

            counter.draw_line(output_frame)
            counter.draw_counts(output_frame)

            if writer is not None:
                writer.write(output_frame)

            if display_enabled:
                try:
                    cv2.imshow("Motion-Aware Smart Surveillance System", output_frame)
                    if config.SHOW_MOTION_MASK:
                        cv2.imshow("Motion Mask", motion_mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as error:
                    display_enabled = False
                    print(f"OpenCV display is unavailable, continuing headless: {error}")

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"Processed {frame_index} frames.")
    if config.SAVE_OUTPUT:
        print(f"Output video saved to: {output_path}")


if __name__ == "__main__":
    process_video()
